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

import os, sys
import argparse, cmd, fcntl, hashlib, json, re, subprocess, time
import timeout_decorator
import pyrosetta as pr
import mszpyrosettaextension as mpre
from pyrosetta.rosetta.core.scoring import all_atom_rmsd
import matplotlib
matplotlib.use("Agg") # otherwise lack of display breaks program
import matplotlib.pyplot as plt

PYROSETTA_ENV = None

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
    """Makes a function repeat forever when continuous mode is enabled, and
    decorates it with times_out."""
    @times_out
    def decorated(slf, *args, **kwargs):
        if slf.settings['continuous_mode']:
            while True:
                f(slf, *args, **kwargs)
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

    It can be asked to log data for any particular calculation performed on any
    particular PDB file. It can also be asked to verify that any particular
    piece of data is up-to-date, and delete it when it isn't. It autonomously
    creates cache files when performing calculations, unless this behavior is
    turned off.
    It is not the buffer's responsibility to keep a directory tree of
    PDBs. That's the problem of whichever subroutine needs to know the directory
    structure.
    The structure of the buffer's data variable is thus:

    { content_hash : [{ pdb_file_path : mtime_at_hashing },
                      { weight_param_1 :
                        { weight_param_2 :
                          { weight_param_3 : ...
                            ... : score } } },
                      { rmsd_type :
                        { other_content_hash : rmsd }}
                      { bound : neighborhood_matrix }] }
    """
    def __init__(self):
        self.cachingp = True
        self.data = {}
    def get_file_essentials(self, path):
        """Returns file's contents, content hash, path, and mtime."""
        contents = None
        with open(path, 'r') as pdb_file:
            contents = pdb_file.read()
        hash_fun = hashlib.md5()
        hash_fun.update(contents)
        content_hash = hash_fun.hexdigest()
        path = os.path.abspath(path) # absolutized
        mtime = os.path.getmtime(path)
        return (contents, content_hash, path, mtime)
    def retrieve_data_from_cache(self, dirpath):
        retrieved_data = None
        with open(os.path.join(dirpath, '.paperclip_cache'), 'r') as cache_file:
            try:
                retrieved_data = json.loads(cache_file.read())
            except ValueError:
                pass
        if retrieved_data:
            def merge_data(buffer_data, new_data):
                pass # TODO
            self.data = merge_data(self.data, retrieved_data)
    @needs_pr_scorefxn
    def calculate_pdb_score(self, contents, params=None):
        global PYROSETTA_ENV
        pose = None
        if params:
            pose = mpre.pose_from_pdbstring_with_params(contents, params)
        else:
            pose = pr.pose_from_pdbstring(contents)
        return PYROSETTA_ENV.scorefxn(pose)
    def update_pdb_score(self, path, params=None):
        contents, content_hash, path, mtime = self.get_file_essentials(path)
        # have we already loaded the cache at this location?
        # TODO
        # create list of args for indexing into score
        # TODO https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
        # do we already have this info?
        # TODO
        # if not, then go ahead and calculate the score, then store it
        score = calculate_pdb_score(contents, params)
        # TODO

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
    data_buffer = {} # contains computed data about PDBs
    text_buffer = '' # contains text output
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
    def do_clear_data_buffer(self, arg):
        """Clear the data buffer of any calculated data:  clear_data_buffer"""
        self.data_buffer = {}
    # Text buffer
    def do_clear_text_buffer(self, arg):
        """Clear the text buffer of any text output:  clear_text_buffer"""
        self.text_buffer = ''
    def do_view_text_buffer(self, arg):
        """View the text buffer, less-style:  view_text_buffer"""
        subprocess.run(['less'], input=bytes(self.text_buffer, 'utf-8'))

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
