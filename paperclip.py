#!/opt/sw/packages/gcc-4.8/python/3.5.2/bin/python3

#### paperclip.py
#### Plotting and Analyzing Proteins Employing Rosetta CLI with PAPERCLIP
#### by Michael Szegedy, 2017 Oct
#### for Khare Lab at Rutgers University

'''This is an interactive suite for generating a variety of plots based on data
obtained from Rosetta and MPRE (mszpyrosettaextension). Mostly the plots
revolve around data that I've had to collect for research, or personal
interest. Most plotting commands are based on Matlab commands.
'''

### Imports

## Python Standard Library
import re
import functools, operator
import hashlib
import argparse, os, time
import subprocess
import json
import cmd
import sys
import fcntl
## Other stuff
import timeout_decorator
import matplotlib
matplotlib.use("Agg") # otherwise lack of display breaks program
import matplotlib.pyplot as plt
import pyrosetta as pr
from pyrosetta.rosetta.core.scoring import all_atom_rmsd
import mszpyrosettaextension as mpre

PYROSETTA_ENV = None
ROSETTA_WEIGHT_NAMES = ['fa_atr',
                        'fa_rep',
                        'fa_sol',
                        'fa_intra_rep',
                        'fa_elec',
                        'pro_close',
                        'hbond_sr_bb',
                        'hbond_lr_bb',
                        'hbond_bb_sc',
                        'hbond_sc',
                        'dslf_fa13',
                        'rama',
                        'omega',
                        'fa_dun',
                        'p_aa_pp',
                        'ref']
ROSETTA_RMSD_TYPES = ['gdtsc',
                      'CA_rmsd',
                      'CA_gdtmm',
                      'bb_rmsd',
                      'bb_rmsd_including_O',
                      'all_atom_rmsd',
                      'nbr_atom_rmsd']

### Decorators

def needs_pr_init(f):
    """Makes sure PyRosetta gets initialized before f is called."""
    def decorated(*args, **kwargs):
        global PYROSETTA_ENV
        if not PYROSETTA_ENV.initp:
            print('PyRosetta not initialized yet. Initializing...')
            pr.init()
            PYROSETTA_ENV.initp = True
        return f(*args, **kwargs)
    return decorated
def needs_pr_scorefxn(f):
    """Makes sure the scorefxn exists before f is called. (In order for the
    scorefxn to exist, pr.init() needs to have been called, so @needs_pr_init
    is unnecessary in front of this.)"""
    def decorated(*args, **kwargs):
        global PYROSETTA_ENV
        if PYROSETTA_ENV.scorefxn is None:
            print('Scorefxn not initialized yet. Initializing from defaults...')
            PYROSETTA_ENV.set_scorefxn()
        return f(*args, **kwargs)
    return decorated
def times_out(f):
    """Makes a function time out after it hits self.timelimit."""
    def decorated(slf, *args, **kwargs):
        if slf.timelimit:
            try:
                return timeout_decorator\
                    .timeout(slf.timelimit)(f)(slf, *args, **kwargs)
            except timeout_decorator.TimeoutError:
                print('Timed out.')
        else:
            return f(slf, *args, **kwargs)
    return decorated
def continuous(f):
    """Makes a function in OurCmdLine repeat forever when continuous mode is
    enabled, and decorates it with times_out."""
    @times_out
    def decorated(slf, *args, **kwargs):
        if slf.settings['continuous_mode']:
            while True:
                f(slf, *args, **kwargs)
        else:
            return f(slf, *args, **kwargs)
    return decorated
def caching(f):
    """Makes a function in a PDBDataBuffer save the cache when finished
    running."""
    def decorated(slf, *args, **kwargs):
        if slf.cachingp:
            retval = f(slf, *args, **kwargs)
            slf.update_caches()
            return retval
        else:
            return f(slf, *args, **kwargs)
    return decorated

### Useful functions
def get_filenames_from_dir_with_extension(dir_path, extension,
                                          strip_extensions_p=False):
    """Returns a list of files from a directory with the path stripped, and
    optionally the extension stripped as well."""
    path_list = str(subprocess.check_output('ls '+os.path.join(dir_path,
                                                               '*'+extension),
                                            shell=True)).split('\\n')
    # Doesn't hurt to validate twice
    stripper = None
    if strip_extensions_p:
        stripper = re.compile(r'[^\\]/([^/]+)' + re.escape(extension) + r'$')
    else:
        stripper = re.compile(r'[^\\]/([^/]+' + re.escape(extension) + r')$')
    # Premature optimization is the root of all evil, but who wants to run
    # the same regex twice?
    return [m.group(1) \
            for m \
            in [stripper.search(path) for path in path_list] \
            if m is not None]
def recursive_dict_merge(a, b, favor_a_p=False):
    """Merges two dicts recursively, favoring the second over the first unless
    otherwise specified."""
    merged = a
    for key in b:
        if key in merged:
            if isinstance(merged[key], dict) and \
               isinstance(b[key], dict):
                merged[key] = recursive_dict_merge(merged[key], b[key],
                                                   favor_a_p)
            elif not isinstance(merged[key], dict) and \
                 not isinstance(b[key], dict):
                merged[key] = a[key] if favor_a_p else b[key]
            else:
                raise ValueError('Incompatible types found in dict'
                                 'merge.')
        else:
            merged[key] = b[key]
    return merged
def get_from_dict_with_list(d, l):
    """Indexes into a multi-layered dict with a list of keywords, one for each
    layer."""
    return functools.reduce(operator.getitem, l, d)
def set_in_dict_with_list(d, l, value):
    """Sets a value inside a multi-layered dict, indexed with a list of
    keywords, one for each layer."""
    for key in l[:-1]:
        d = d.setdefault(key, {})
    d[l[-1]] = value

### Housekeeping classes

class PyRosettaEnv():
    """Stores stuff relating to PyRosetta, like whether pr.init() has been
    called."""
    def __init__(self):
        self.initp = False
        self.scorefxn = None
    @needs_pr_init
    def set_scorefxn(self, name=None, patch=None):
        """Sets the scorefxn and optionally applies a patch. Defaults to
        get_fa_scorefxn() with no arguments."""
        if name is None:
            self.scorefxn = pr.get_fa_scorefxn()
        else:
            self.scorefxn = pr.create_score_function(name)
        if patch:
            self.scorefxn.apply_patch_from_file(patch)

class PDBDataBuffer():
    """Singleton class that stores information about PDBs, to be used as this
    program's data buffer. Dict-reliant, so stringly typed. It can store the
    following things:

      - Contents hashes (in case two PDBs are identical); this is how it indexes
        its data
      - List of paths to PDB files that have the corresponding content hash
      - List of mtimes for each file when its content hash was computed
      - Scores for any combination of weights
      - RMSD vs protein with another content hash
      - Residue neighborhood matrices for arbitrary bounds
      - What caches it loaded to create this buffer

    It can be asked to log data for any particular calculation performed on any
    particular PDB file. It can also be asked to verify that any particular
    piece of data is up-to-date, and delete it when it isn't. It autonomously
    creates cache files when performing calculations, unless this behavior is
    turned off.
    It is not the buffer's responsibility to keep a directory tree of
    PDBs. That's the problem of whichever subroutine needs to know the directory
    structure.
    The structure of the buffer's data variable is thus:

    { content_key : [{ pdb_file_path : mtime_at_hashing },
                     { weight_param_1 :
                       { weight_param_2 :
                         { weight_param_3 : ...
                           ... : score } } },
                     { rmsd_type :
                       { other_content_key : rmsd } }
                     { bound : neighborhood_matrix }] }
    """
    ## Core functionality
    def __init__(self):
        self.cachingp = True
        self.data = {}
    def merge_new_data(self, new_data):
        """Merges a data dict for a PDBDataBuffer into this one. The new dict
        takes precence over the old one unless otherwise specified."""
        for content_key in new_data:
            if content_key not in self.data.keys():
                self.data[content_key] = new_data[content_key]
            else:
                for index, entry in enumerate(new_data[content_key]):
                    if index == 0:
                        for path in entry:
                            try:
                                our_mtime = self.data[content_key][0][path]
                                new_mtime = entry[path]
                                newest_mtime = \
                                    our_mtime if our_mtime > new_mtime \
                                    else new_mtime
                                self.data[content_key][0][path] = newest_mtime
                            except KeyError:
                                self.data[content_key][0][path] = entry[path]
                    else:
                        self.data[content_key][index] = \
                            recursive_dict_merge(self.data[content_key][index],
                                                 entry)
    def retrieve_data_from_cache(self, dirpath):
        """Retrieves data from a cache file of a directory. The data in the
        file is a JSON of a data dict of a PDBDataBuffer, except instead of
        file paths, it has just filenames."""
        retrieved_data = None
        try:
            with open(os.path.join(dirpath, '.paperclip_cache'), 'r') as cache_file:
                try:
                    retrieved_data = json.loads(cache_file.read())
                except ValueError:
                    raise FileNotFoundError('No cache file found.')
            if retrieved_data:
                for content_key in retrieved_data:
                    paths = retrieved_data[content_key][0]
                    for path in paths:
                        retrieved_data[content_key]\
                                      [0][os.path.abspath(path)] = \
                            retrieved_data[content_key][0][path]
                        del retrieved_data[content_key][0][path]
                self.data = self.merge_new_data(retrieved_data)
        except FileNotFoundError:
            pass
    def update_caches(self):
        """Updates the caches for every directory the cache knows about."""
        # Basically we need to reorganize the data structure so that stuff is
        # organized by directory first, then content key, then filename.
        new_dict = {}
        for content_key, content_value in self.data.items():
            for path in content_value[0]:
                new_dict_key = os.path.dirname(path)
                filename = os.path.basename(path)
                if new_dict_key in new_dict:
                    if content_key in new_dict[new_dict_key] and \
                       filename not in new_dict[new_dict_key][content_key]:
                        new_dict[new_dict_key][content_key][0][filename] = \
                            content_value[0][filename]
                    else:
                       to_add = self.data[content_key]
                       to_add[0] = {filename: content_value[0][filename]}
                       new_dict[new_dict_key][content_key] = to_add
                else:
                    to_add = {content_key: self.data[content_key]}
                    to_add[content_key][0] = [filename]
                    new_dict[new_dict_key] = to_add
        for dir_path, new_data in new_dict.items():
            cache_filename = os.path.join(dir_path, '.paperclip_cache')
            with open(cache_filename, 'w+') as cache_file:
                # try to get lock on cache file
                while True:
                    try:
                        fcntl.flock(cache_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.05)
                # now that we have a lock, load the newest version of the cache
                cache_data = None
                try:
                    cache_data = json.loads(cache_file.read())
                except ValueError:
                    cache_data = {}
                merged_cache = PDBDataBuffer()
                merged_cache.data = cache_data
                merged_cache.merge_new_data(new_data)
                cache_file.write(json.dumps(merged_cache.data))
                fcntl.flock(cache_file, fcntl.LOCK_UN)
    def get_pdb_file_essentials(self, path):
        """Returns pdb file's contents, content hash, path, and mtime. As a
        side effect, also updates the data dict with the info of the file."""
        contents = None
        with open(path, 'r') as pdb_file:
            contents = pdb_file.read()
        hash_fun = hashlib.md5()
        hash_fun.update(contents.encode())
        content_key = hash_fun.hexdigest()
        path = os.path.abspath(path) # absolutized
        mtime = os.path.getmtime(path)
        self.data.setdefault(content_key, [{},{},{},{}])
        print(self.data)
        print(content_key)
        self.data[content_key][0].setdefault(path, mtime)
        return (contents, content_key, path, mtime)

    ## Updating Rosetta stuff
    # Scoring
    @needs_pr_scorefxn
    def get_score_indices_list(self):
        """Builds a list of indices to index score with."""
        global PYROSETTA_ENV
        return (str(scorefxn.get_weight(
                        getattr(pr.rosetta.core.scoring, name))) \
                for name in ROSETTA_WEIGHT_NAMES)
    @caching
    @needs_pr_scorefxn
    def calculate_and_update_pdb_score(self, contents, content_key,
                                       params=None):
        """Calculates the score of a protein based on the provided contents of
        its PDB file."""
        global PYROSETTA_ENV
        pose = None
        if params:
            pose = mpre.pose_from_pdbstring_with_params(contents, params)
        else:
            pose = pr.pose_from_pdbstring(contents)
        indices = self.get_score_indices_list()
        set_in_dict_with_list(self.data[content_key][1],
                              indices, PYROSETTA_ENV.scorefxn(pose))
    def update_pdb_score(self, path, params=None):
        contents, content_key, path, mtime = \
            self.get_pdb_file_essentials(path)
        self.retrieve_data_from_cache(os.path.dirname(path))
        try:
            if self.data[content_key][0][path] >= mtime and \
               get_from_dict_with_list(self.data[content_key][1],
                                       self.get_score_indices_list()):
                # no need to update
                return
        except KeyError:
            self.calculate_and_update_pdb_score(contents, params)
    def get_pdb_score_from_path(self, path, params=None):
        contents, content_key, path, mtime = \
            self.get_pdb_file_essentials(path)
        self.update_pdb_score(path, params)
        indices = self.get_score_indices_list()
        return get_from_dict_with_list(self.data[content_key][1], indices)
    # RMSDing
    @caching
    @needs_pr_init
    def calculate_and_update_pdb_rmsd(self, contents_lhs, content_key_lhs,
                                      contents_rhs, content_key_rhs,
                                      rmsd_type, params=None):
        """Calculates the RMSD of two proteins from each other and stores
        it, without assuming commutativity."""
        pose_lhs = None
        if params:
            pose_lhs = mpre.pose_from_pdbstring_with_params(contents_lhs,
                                                            params)
        else:
            pose_lhs = pr.pose_from_pdbstring(contents_lhs)
        pose_rhs = None
        if params:
            pose_rhs = mpre.pose_from_pdbstring_with_params(contents_rhs,
                                                            params)
        else:
            pose_rhs = pr.pose_from_pdbstring(contents_rhs)
        self.data[content_key_lhs][2].setdefault(rmsd_type,{})
        self.data[content_key_lhs][2][rmsd_type].setdefault(content_key_rhs,{})
        self.data[content_key_lhs][2][rmsd_type][content_key_rhs] = \
            getattr(pr.rosetta.core.scoring, rmsd_type)(pose_lhs, pose_rhs)
    def update_pdb_rmsd(self, path_lhs, path_rhs, rmsd_type, params=None):
        contents_lhs, content_key_lhs, path_lhs, mtime_lhs = \
            self.get_pdb_file_essentials(path_lhs)
        self.retrieve_data_from_cache(os.path.dirname(path_lhs))
        contents_rhs, content_key_rhs, path_rhs, mtime_rhs = \
            self.get_pdb_file_essentials(path_rhs)
        self.retrieve_data_from_cache(os.path.dirname(path_rhs))
        try:
            if self.data[content_key_lhs][0][path_lhs] >= mtime_lhs and \
               self.data[content_key_lhs][2][rmsd_type][content_key_rhs]:
                # no need to update
                return
        except KeyError:
            self.calculate_and_update_pdb_rmsd(contents_lhs, content_key_lhs,
                                               contents_rhs, content_key_rhs,
                                               rmsd_type, params)
    def get_pdb_rmsd_from_path(self, path_lhs, path_rhs, rmsd_type,
                               params=None):
        contents_lhs, content_key_lhs, path_lhs, mtime_lhs = \
            self.get_pdb_file_essentials(path_lhs)
        contents_rhs, content_key_rhs, path_rhs, mtime_rhs = \
            self.get_pdb_file_essentials(path_rhs)
        self.update_pdb_rmsd(path_lhs, path_rhs, rmsd_type, params)
        return self.data[content_key_lhs][2][rmsd_type]
    # Finding neighborhood matrices
    @caching
    @needs_pr_init
    def calculate_and_update_pdb_neighbors(self, contents, content_key,
                                           bound=None, params=None):
        """Calculates the residue neighborhood matrix of a protein based on the
        provided contents of its PDB file."""
        pose = None
        if params:
            pose = mpre.pose_from_pdbstring_with_params(contents, params)
        else:
            pose = pr.pose_from_pdbstring(contents)
        result = []
        n_residues = pose.size()
        for i in range(n_residues):
            result.append([])
            for j in range(n_residues):
                result[-1].append(mpre.res_neighbors_p(pose,i,j,bound=bound))
        self.data[content_key][3][str(bound)] = result
    def update_pdb_neighbors(self, path, bound=None, params=None):
        contents, content_key, path, mtime = \
            self.get_pdb_file_essentials(path)
        self.retrieve_data_from_cache(os.path.dirname(path))
        try:
            if self.data[content_key][0][path] >= mtime and \
               self.data[content_key][3][str(bound)]:
                # no need to update
                return
        except KeyError:
            self.calculate_and_update_pdb_neighbors(contents, bound, params)
    def get_pdb_neighbors_from_path(self, path, bound, params=None):
        contents, content_key, path, mtime = \
            self.get_pdb_file_essentials(path)
        self.update_pdb_neighbors(path, bound, params)
        return self.data[content_key][3][str(bound)]

### Main class

class OurCmdLine(cmd.Cmd):
    """Singleton class for our interactive CLI."""
    ## Built-in vars
    intro = 'Welcome to PAPERCLIP. Type help or ? to list commands.'
    prompt = '* '
    ## Our vars (don't wanna mess with __init__)
    cmdfile = None
    settings = {'calculation': True,
                'caching': True,
                'plotting': True,
                'recalculate_energies': True,
                'continuous_mode': False}
    timelimit = 0
    ## The two buffers:
    data_buffer = PDBDataBuffer() # contains computed data about PDBs
    text_buffer = ''              # contains text output
    # There is also a plot buffer, but that is contained within pyplot.

    ## Housekeeping
    def do_quit(self, arg):
        """Stop recording and exit:  quit"""
        self.close()
        return True
    def do_bye(self, arg):
        """Stop recording and exit:  bye"""
        return self.do_quit(arg)
    def do_EOF(self, arg):
        """Stop recording and exit:  EOF  |  ^D"""
        return self.do_quit(arg)
    def do_exit(self, arg):
        """Stop recording and exit:  exit"""
        return self.do_quit(arg)
    def emptyline(self):
        pass

    ## Parsing
    def get_arg_position(self, text, line):
        """For completion; gets index of current positional argument (returns 1 for
        first arg, 2 for second arg, etc.)."""
        return len(line.split()) - (text != '')

    ## Recording and playing back commands
    def do_record(self, arg):
        """Save future commands to filename:  record plot.cmd"""
        self.cmdfile = open(arg, 'w')
    def do_playback(self, arg):
        """Play back commands from a file:  playback plot.cmd"""
        self.close()
        with open(arg) as f:
            self.cmdqueue.extend(f.read().splitlines())
    def precmd(self, line):
        if self.cmdfile and 'playback' not in line:
            print(line, file=self.cmdfile)
        return line
    def close(self):
        if self.cmdfile:
            self.cmdfile.close()
            self.cmdfile = None

    ## Shell stuff
    def do_shell(self, arg):
        """Call a shell command:  shell cd dir  |  !cd dir"""
        os.system(arg)
    def do_cd(self, arg):
        """Change the current working directory:  cd dir"""
        try:
            os.chdir(arg)
        except FileNotFoundError:
            print('No such file or directory: ' + arg)
    def do_mv(self, arg):
        """Call the shell command mv:  mv a b"""
        self.do_shell('mv ' + arg)
    def do_rm(self, arg):
        """Call the shell command rm:  rm a"""
        self.do_shell('rm ' + arg)
    def do_ls(self, arg):
        """Call the shell command ls:  ls .."""
        self.do_shell('ls ' + arg)
    def do_pwd(self, arg):
        """Get the current working directory:  pwd"""
        os.getcwd()

    ## Settings
    def do_get_settings(self, arg):
        """Print settings of current session:  get_settings"""
        for key, value in self.settings.items():
            transformed_value = 'yes' if value == True else \
                                'no'  if value == False else value
            print('{0:<20}{1:>8}'.format(key+':', transformed_value))
    def do_set(self, arg):
        """Set or toggle a yes/no setting variable in the current session:
  set calculation no  |  set calculation

Available settings are:
  caching: Whether to cache the results of calculations or not.
  calculation: Whether to perform new calculations for values that may be
      outdated in the cache, or just use the possibly outdated cached values.
      Turning this off disables caching, even if 'caching' is set to 'yes'.
  continuous_mode: Repeat all analysis and plotting commands until they hit the
      time limit, or forever. Useful if a program is still generating data for
      a directory, but you want to start caching now. (To set a time limit, use
      the command 'set_timelimit'.)
  plotting: Whether to actually output plots, or to just perform and cache the
      calculations for them. Disabling both this and 'calculation' makes most
      analysis and plotting commands do nothing.
  recalculate_energies: Recalculate energies found in cache when plotting. This
      may be necessary if they were computed with a custom scorefxn that wasn't
      recorded in the cache file."""
        args = arg.split()
        varname  = None
        varvalue = None
        if len(args) == 2:
            varname  = args[0]
            varvalue = args[1]
        elif len(args) == 1:
            varname = args[0]
        else:
            print('Incorrect command usage. Try \'help set\'.')
        if varvalue is None:
            try:
                self.settings[varname] = not self.settings[varname]
            except KeyError:
                print('That\'s not a valid setting name. Try '
                      '\'get_settings\'.')
                return
        else:
            value = None
            if str.lower(varvalue) in ('yes','y'):
                value = True
            elif str.lower(varvalue) in ('no','n'):
                value = False
            else:
                print('That\'s not a valid setting value. Try \'yes\' or'
                      '\'no\'.')
                return
            if varname in self.settings.keys():
                self.settings[varname] = value
            else:
                print('That\'s not a valid setting name. Try '
                      '\'get_settings\'.')
                return
    def complete_set(self, text, line, begidx, endidx):
        position = self.get_arg_position(text, line)
        if position == 1:
            return [i for i in list(self.settings.keys()) if i.startswith(text)]
        elif position == 2:
            return [i for i in ['yes', 'no'] if i.startswith(text)]
    def do_get_timelimit(self, arg):
        """Print the current time limit set on analysis commands:
  get_timelimit"""
        print(str(self.timelimit)+' seconds')
    def do_set_timelimit(self, arg):
        """Set a time limit on analysis commands, in seconds. Leave as 0 to let
commands run indefinitely:
   set_timelimit 600"""
        try:
            self.timelimit = int(arg)
        except ValueError:
            print("Enter an integer value of seconds.")

    ## Buffer interaction
    # Data buffer
    def do_clear_data(self, arg):
        """Clear the data buffer of any data:  clear_data"""
        self.data_buffer = PDBDataBuffer()
    # Text buffer
    def do_clear_text(self, arg):
        """Clear the text buffer of any text output:  clear_text"""
        self.text_buffer = ''
    def do_view_text(self, arg):
        """View the text buffer, less-style:  view_text"""
        subprocess.run(['less'], input=bytes(self.text_buffer, 'utf-8'))
    # Plot buffer
    def do_clear_plot(self, arg):
        """Clear the plot buffer:  clear_plot"""
        plt.cla()

    ## Basic Rosetta stuff
    def do_get_scorefxn(self, arg):
        """Print the name of the current scorefxn, if any:  get_scorefxn"""
        global PYROSETTA_ENV
        if PYROSETTA_ENV.scorefxn is None:
            print('No scorefxn currently set.')
        else:
            print(PYROSETTA_ENV.scorefxn.get_name())
    def do_set_scorefxn(self, arg):
        """Set the current scorefxn, optionally applying a patchfile:
    set_scorefxn ref2015  |  set_scorefxn ref2015 docking"""
        global PYROSETTA_ENV
        args = arg.split()
        name = None
        patch = None
        if len(args) == 2:
            name = args[0]
            patch = args[1]
        elif len(args) == 1:
            name = args[0]
        else:
            print('Incorrect command usage. Try \'help set_scorefxn\'')
        if name.startswith('talaris'):
            print('Setting the scorefxn to a talaris flavor crashes Rosetta. '
                  'Don\'t do that.')
            return
        PYROSETTA_ENV.set_scorefxn(name=name, patch=patch)
    def complete_set_scorefxn(self, text, line, begidx, endidx):
        scorefxn_list = get_filenames_from_dir_with_extension(
            '$HOME/.local/lib/python3.5/site-packages/pyrosetta*/pyrosetta/'
            'database/scoring/weights/',
            '.wts', strip_extensions_p = True)
        patches_list = get_filenames_from_dir_with_extension(
            '$HOME/.local/lib/python3.5/site-packages/pyrosetta*/pyrosetta/'
            'database/scoring/weights/',
            '.wts_patch', strip_extensions_p = True)
        position = self.get_arg_position(text, line)
        if position == 1 :
            return [i for i in scorefxn_list if i.startswith(text)]
        elif position == 2:
            return [i for i in patches_list if i.startswith(text)]

    ## Plots
    def do_plot_title(self, arg):
        """Set the title of the current plot:  plot_title My title"""
        plt.title(arg)
    def do_plot_xlabel(self, arg):
        """Set the xlabel of the current plot:  plot_xlabel My xlabel"""
        plt.xlabel(arg)
    def do_plot_ylabel(self, arg):
        """Set the ylabel of the current plot:  plot_ylabel My ylabel"""
        plt.ylabel(arg)
    def do_save_plot(self, arg):
        """Save the plot currently in the plot buffer:  save_plot"""
        plt.savefig(arg, format=arg.split('.')[-1].lower())
    def do_plot_dir_rmsd_vs_score(self, arg):
        """For each PDB in a directory, plot the RMSDs vs a particular file
against their energy score:
    plot_dir_rmsd_vs_score indir infile.pdb  |
    plot_dir_rmsd_vs_score indir infile.pdb --params ABC  |
    plot_dir_rmsd_vs_score indir infile.pdb 4.6"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--in_dir',
                            dest='in_dir',
                            action='store')
        parser.add_argument('--in_file',
                            dest='in_file',
                            action='store')
        parser.add_argument('bound',
                            action='store',
                            nargs='?',
                            default=None)
        parser.add_argument('--style',
                            dest='style',
                            action='store',
                            default='ro')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs=argparse.REMAINDER,
                            default=None)
        parsed_args = parser.parse_args(arg.split())
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        data = ((self.data_buffer.get_pdb_rmsd_from_path(
                     parsed_args.in_file,
                     os.path.join(parsed_args.in_dir, filename),
                     'all_atom_rmsd', parsed_args.params),
                 self.data_buffer.get_pdb_score_from_path(
                     os.path.join(parsed_args.in_dir,
                                  filename), parsed_args.params)) \
                for filename in filenames_in_in_dir \
                if filename.endswith('.pdb'))
        data = (datapoint for datapoint in data \
                if datapoint[1] < 0)
        rmsds  = [datapoint[0] for datapoint in data]
        scores = [datapoint[1] for datapoint in data]
        plt.plot(rmsds, scores, parsed_args.style)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Interactive command-line interface for plotting and "
                      "analysis of batches of PDB files.")
    parser.add_argument("--background",
                        dest="backgroundp",
                        action="store_true",
                        help="Run given script in background and then "
                             "terminate. If no script is given, just do "
                             "nothing and terminate.")
    parser.add_argument("--continuous",
                        dest="continuousp",
                        action="store_true",
                        help="Re-run caching operations until they hit the "
                             "time limit or forever. By default, suppresses "
                             "plots.")
    parser.add_argument("script",
                        action="store",
                        nargs='?',
                        default=None,
                        help=".cmd file to run before entering interactive "
                             "mode.")
    parsed_args = parser.parse_args()
    PYROSETTA_ENV = PyRosettaEnv()
    OURCMDLINE = OurCmdLine()
    OURCMDLINE.settings['continuous_mode'] = parsed_args.continuousp
    OURCMDLINE.settings['plotting'] = not parsed_args.continuousp
    if parsed_args.script is not None:
        OURCMDLINE.do_playback(parsed_args.script)
    if not parsed_args.backgroundp:
        OurCmdLine().cmdloop()
