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
print('HELLO BEFORE IMPORTS')
## Python Standard Library
import re
import types, copy
import functools, operator
import hashlib
import argparse, os, time
import subprocess
import cmd
import sys
import ast
## Other stuff
import timeout_decorator
from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use("Agg") # otherwise lack of display breaks program
import matplotlib.pyplot as plt
import pyrosetta as pr
import mszpyrosettaextension as mpre

STDOUT = sys.stdout;
MPICOMM   = MPI.COMM_WORLD
MPIRANK   = MPICOMM.Get_rank()
MPISIZE   = MPICOMM.Get_size()
MPISTATUS = MPI.Status()
PYROSETTA_ENV = None
ROSETTA_RMSD_TYPES = ['gdtsc',
                      'CA_rmsd',
                      'CA_gdtmm',
                      'bb_rmsd',
                      'bb_rmsd_including_O',
                      'all_atom_rmsd',
                      'nbr_atom_rmsd']
print('HELLO AFTER IMPORTS',flush=True)

### Decorators

def needs_pr_init(f):
    """Makes sure PyRosetta gets initialized before f is called."""
    @functools.wraps(f)
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
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        global PYROSETTA_ENV
        if PYROSETTA_ENV.scorefxn is None:
            print('Scorefxn not initialized yet. Initializing from defaults...')
            PYROSETTA_ENV.set_scorefxn()
        return f(*args, **kwargs)
    return decorated
def times_out(f):
    """Makes a function time out after it hits self.timelimit."""
    @functools.wraps(f)
    def decorated(self_, *args, **kwargs):
        if self_.timelimit:
            try:
                return timeout_decorator\
                    .timeout(self_.timelimit)(f)(self_, *args, **kwargs)
            except timeout_decorator.TimeoutError:
                print('Timed out.')
        else:
            return f(self_, *args, **kwargs)
    return decorated
def continuous(f):
    """Makes a function in OurCmdLine repeat forever when continuous mode is
    enabled, and decorates it with times_out."""
    @functools.wraps(f)
    @times_out
    def decorated(self_, *args, **kwargs):
        if self_.settings['continuous_mode']:
            while True:
                f(self_, *args, **kwargs)
        else:
            return f(self_, *args, **kwargs)
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

def make_PDBDataBuffer_accessor(data_name):
    """Create an accessor function for a PDBDataBuffer data type."""
    # Python duck typing philosophy says we shouldn't do any explicit
    # checking of the thing stored at the attribute, but in any case,
    # it should be a function that takes a pdb_contents string as its
    # first argument, a params list as a keyword argument, and a bunch
    # of hashable arguments as the rest of its arguments.
    # TODO: Fix this so that you don't need to pass the scorefxn hash
    #       to get_score().
    # TODO: Fix this so that calculate_rmsd() can take a contents
    #       hash, rather than a path.
    def accessor(self_, file_path, *args, params=None, **kwargs):
        self_.retrieve_data_from_cache(os.path.dirname(file_path))
        file_info = self_.get_pdb_file_info(file_path)
        file_data = self_.data.setdefault(file_info.hash, {}) \
                              .setdefault(data_name, {})
        kwargs_args_list = list(kwargs.keys())
        kwargs_args_list.sort()
        accessor_tuple = tuple(args) + \
                         tuple(kwargs[arg] for arg in kwargs_args_list)
        try:
            return file_data[accessor_tuple]
        except KeyError:
            file_data[accessor_tuple] = \
                getattr(self_, 'calculate_'+data_name) \
                    (file_info.get_contents(), *args,
                     params=params, **kwargs)
            if self_.cachingp:
                self_.update_caches()
            return file_data[accessor_tuple]
    accessor.__name__ = 'get_'+data_name
    accessor.__doc__  = 'Call calculate_'+data_name+ \
                        '() on a file path with caching magic.'
    return accessor

def make_PDBDataBuffer_gatherer(data_name):
    """Make an accessor for a PDBDataBuffer that concurrently operates on a
    list of filenames instead of a single filename."""
    def gatherer(self_, file_list, *args, params=None, **kwargs):
        abs_file_list = [os.path.abspath(path) for path in file_list]
        if MPISIZE == 1:
            return [getattr(self_, 'get_'+data_name) \
                        (file_path, *args,
                         params=params, **kwargs) \
                    for file_path in abs_file_list]
        else:
            # getting this instead of a path makes a worker thread die:
            QUIT_GATHERING_SIGNAL = -1
            READY_TAG = 0 # used once by worker, to sync startup
            DONE_TAG  = 1 # used by worker to pass result and request new job
            WORK_TAG  = 2 # used by master to pass next path to worker
            # will be used to index into self_.data to retrieve final result:
            file_indices_dict = {}
            print('\n\nretrieving caches for for the first time\n\n', flush=True)
            for dirname in set((os.path.dirname(path) \
                                for path in abs_file_list)):
                self_.retrieve_data_from_cache(dirname)
            if MPIRANK == 0:
                current_file_index = 0
                working_threads = MPISIZE-1
                def save_result(result):
                    file_info_dict = result[0]
                    file_data      = result[1]
                    accessor_tuple = result[2]
                    self_.pdb_paths[file_info_dict['path']] = \
                        {'mtime': file_info_dict['mtime'],
                         'hash':  file_info_dict['hash']}
                    self_.data.setdefault(file_info_dict['hash'], {}) \
                              .setdefault(data_name, {}) \
                              [accessor_tuple] = file_data
                    self_.update_caches()
                    file_indices_dict[file_info_dict['path']] = \
                        (file_info_dict['hash'], data_name, accessor_tuple)
                    print('\n\njust saved result for '+file_info_dict['path']+'\n\n', flush=True)
                    sys.stdout.flush()
                while True:
                    print('\n\nabout to try to receive a message\n\n', flush=True)
                    result = MPICOMM.recv(source=MPI.ANY_SOURCE,
                                          tag=MPI.ANY_TAG,
                                          status=MPISTATUS)
                    print('\n\nmessage received\n\n', flush=True)
                    result_source = MPISTATUS.Get_source()
                    result_tag    = MPISTATUS.Get_tag()
                    if current_file_index < len(abs_file_list):
                        print('\n\nabout to try to send a WORK message\n\n', flush=True)
                        MPICOMM.send(abs_file_list[current_file_index],
                                     dest=result_source, tag=WORK_TAG)
                        print('\n\nWORK message sent\n\n', flush=True)
                        current_file_index += 1
                    else:
                        print('\n\nabout to try to send a QUIT message\n\n', flush=True)
                        MPICOMM.send(QUIT_GATHERING_SIGNAL,
                                     dest=result_source, tag=WORK_TAG)
                        print('\n\nQUIT message sent\n\n', flush=True)
                        working_threads -= 1
                    if result_tag == DONE_TAG:
                        save_result(result)
                    if working_threads <= 0:
                        break
            else:
                MPICOMM.send(None, dest=0, tag=READY_TAG)
                while True:
                    file_path = MPICOMM.recv(source=0, tag=WORK_TAG)
                    if file_path == QUIT_GATHERING_SIGNAL:
                        break
                    # this is nearly a copy/paste of the accessor code, but
                    # it stores some extra stuff and doesn't retrieve or write
                    # any caches
                    result_data = None
                    file_info = self_.get_pdb_file_info(file_path)
                    file_data = self_.data.setdefault(file_info.hash, {}) \
                                          .setdefault(data_name, {})
                    kwargs_args_list = list(kwargs.keys())
                    kwargs_args_list.sort()
                    accessor_tuple = tuple(args) + \
                                     tuple(kwargs[arg] \
                                           for arg in kwargs_args_list)
                    try:
                        result_data = file_data[accessor_tuple]
                    except KeyError:
                        result_data = \
                            getattr(self_, 'calculate_'+data_name) \
                                (file_info.get_contents(), *args,
                                 params=params, **kwargs)
                    # can't be sending an object with a pointer in it to an
                    # object in a different thread:
                    file_info.pdb_paths_dict = None
                    file_info_dict = {'path':  file_info.path,
                                      'mtime': file_info.mtime,
                                      'hash':  file_info.hash}
                    MPICOMM.send(copy.deepcopy([file_info_dict,
                                                result_data,
                                                accessor_tuple]),
                                 dest=0, tag=DONE_TAG)
            # Synchronize everything that could have possibly changed:
            self_.data = MPICOMM.bcast(self_.data, root=0)
            self_.pdb_paths = MPICOMM.bcast(self_.pdb_paths, root=0)
            self_.cache_paths = MPICOMM.bcast(self_.cache_paths, root=0)
            file_indices_dict = MPICOMM.bcast(file_indices_dict, root=0)
            # All threads should be on the same page at this point.
            retlist = []
            for path in abs_file_list:
                indices = file_indices_dict[path]
                retlist.append(self_.data[indices[0]][indices[1]][indices[2]])
            return retlist
    gatherer.__name__ = 'gather_'+data_name
    gatherer.__doc__  = 'Call calculate_'+data_name+ \
                        '() on a list of file paths with concurrency magic.'
    return gatherer

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
    @needs_pr_scorefxn
    def get_scorefxn_hash(self):
        """Gets a hash of the properties of the current scorefxn."""
        hash_fun = hashlib.md5()
        weights = self.scorefxn.weights()
        hash_fun.update(weights.weighted_string_of(weights).encode())
        return hash_fun.hexdigest()

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
    The structure of the buffer's data variables are thus:

    self.data:
    { content_key : { data_function_name :
                        { data_function_params_tuple : data } } }
    self.pdb_paths:
    { pdb_file_path : { 'mtime' : mtime_at_last_hashing,
                        'hash'  : contents_hash } }
    self.cache_paths:
    { cache_file_dir_path : mtime_at_last_access }
    """
    ## Core functionality
    def __init__(self):
        print('\n\nconstructing PDBDataBuffer\n\n', flush=True)
        self.cachingp = MPIRANK == 0
        self.data = {}
        self.pdb_paths = {}
        self.cache_paths = {}
        # Monkey patch in the data accessors:
        attribute_names = dir(self)
        for data_name in (attribute_name[10:] \
                          for attribute_name in attribute_names \
                          if attribute_name.startswith('calculate_')):
            setattr(self, 'get_'+data_name,
                    types.MethodType(make_PDBDataBuffer_accessor(data_name),
                                     self))
            setattr(self, 'gather_'+data_name,
                    types.MethodType(make_PDBDataBuffer_gatherer(data_name),
                                     self))
    def retrieve_data_from_cache(self, dirpath, cache_fd=None):
        """Retrieves data from a cache file of a directory. The data in the
        file is a JSON of a data dict of a PDBDataBuffer, except instead of
        file paths, it has just filenames."""
        absdirpath = os.path.abspath(dirpath)
        try:
            cache_path = os.path.join(absdirpath, '.paperclip_cache')
            diskmtime = os.path.getmtime(cache_path)
            ourmtime = self.cache_paths.setdefault(absdirpath, 0)
            if diskmtime > ourmtime:
                retrieved = None
                if cache_fd is not None:
                    pos = cache_fd.tell()
                    cache_fd.seek(0)
                    try:
                        retrieved = ast.literal_eval(cache_fd.read())
                    except SyntaxError:
                        pass
                    cache_fd.seek(pos)
                else:
                    try:
                        with open(cache_path, 'r') as cache_file:
                            retrieved = ast.literal_eval(cache_file.read())
                    except (FileNotFoundError, SyntaxError):
                        pass
                if retrieved is not None:
                    diskdata, disk_pdb_info = retrieved
                    for content_key, content in diskdata.items():
                        for data_name, data_keys in content.items():
                            for data_key, data in data_keys.items():
                                self.data.setdefault(content_key,{})
                                self.data[content_key].setdefault(data_name,{})
                                self.data[content_key] \
                                         [data_name] \
                                         [data_key] = data
                    for pdb_name, pdb_info_pair in disk_pdb_info.items():
                        pdb_path = os.path.join(absdirpath, pdb_name)
                        ourpdb_info_pair = self.pdb_paths.get(pdb_path, None)
                        ourpdbmtime = None
                        if ourpdb_info_pair is not None:
                            ourpdbmtime = ourpdb_info_pair['mtime']
                        else:
                            ourpdbmtime = 0
                        if pdb_info_pair['mtime'] > ourpdbmtime:
                            self.pdb_paths[pdb_path] = pdb_info_pair
            self.cache_paths[absdirpath] = diskmtime
        except FileNotFoundError:
            pass
    def update_caches(self):
        """Updates the caches for every directory the cache knows about. Note:
        this method is not concurrency-safe. It is your job to write
        concurrent code around it."""
        dir_paths = {}
        for pdb_path in self.pdb_paths.keys():
            dir_path, pdb_name = os.path.split(pdb_path)
            dir_paths.setdefault(dir_path, set())
            dir_paths[dir_path].add(pdb_name)
        for dir_path, pdb_names in dir_paths.items():
            cache_path = os.path.join(dir_path, '.paperclip_cache')
            cache_file = None
            try:
                cache_file = open(cache_path, 'r+')
            except FileNotFoundError:
                cache_file = open(cache_path, 'w+')
            self.retrieve_data_from_cache(dir_path, cache_fd=cache_file)
            dir_data = {}
            dir_pdb_paths = {}
            for pdb_name in pdb_names:
                pdb_path = os.path.join(dir_path, pdb_name)
                dir_pdb_paths[pdb_name] = self.pdb_paths[pdb_path]
                ourhash = dir_pdb_paths[pdb_name]['hash']
                try:
                    dir_data[ourhash] = self.data[ourhash]
                except KeyError:
                    pass
            cache_file.write(str([dir_data, dir_pdb_paths]))
            cache_file.close()
            self.cache_paths[dir_path] = os.path.getmtime(cache_path)
    def get_pdb_file_info(self, path):
        """Returns an object that in theory contains the absolute path, mtime, contents
        hash, and maybe contents of a PDB. The first three are accessed
        directly, while the last is accessed via an accessor method, so that it
        can be retrieved if necessary. Creating and updating the object both
        update the external PDBDataBuffer's info on that pdb.
        """
        class PDBFileInfo():
            def __init__(self_, path_):
                self_.path = os.path.abspath(path_)
                self_.pdb_paths_dict = self.pdb_paths.setdefault(self_.path, {})
                self_.mtime = self_.pdb_paths_dict.setdefault('mtime', 0)
                self_.hash = self_.pdb_paths_dict.setdefault('hash', None)
                self_.contents = None
                diskmtime = os.path.getmtime(self_.path)
                if diskmtime > self_.mtime or self_.hash is None:
                    self_.update(diskmtime)
            def update(self_, mtime = None):
                diskmtime = mtime or os.path.getmtime(self_.path)
                if diskmtime > self_.mtime or self_.contents is None:
                    with open(self_.path, 'r') as pdb_file:
                        self_.contents = pdb_file.read()
                    hash_fun = hashlib.md5()
                    hash_fun.update(self_.contents.encode())
                    self_.hash = hash_fun.hexdigest()
                    self_.mtime = diskmtime
                    self_.pdb_paths_dict['hash'] = self_.hash
                    self_.pdb_paths_dict['mtime'] = self_.mtime
            def get_contents(self_, mtime = None):
                self_.update(mtime)
                return self_.contents
        return PDBFileInfo(path)

    ## Auxiliary stuff
    @needs_pr_init
    def get_pdb_contents_pose(self, contents, params=None):
        """Returns a Pose of a given pdb file contents string."""
        pose = None
        if params:
            pose = mpre.pose_from_pdbstring_with_params(contents, params)
        else:
            pose = pr.Pose()
            pr.rosetta.core.import_pose.pose_from_pdbstring(pose, contents)
        return pose

    ## Calculating Rosetta stuff
    # Each calculate_<whatever> also implicity creates a get_<whatever>, which
    # is just the same thing but with all the buffer/caching magic attached, 
    # and with the contents arg replaced with a path arg. It also creates a
    # gather_<whatever>, which outwardly looks like get_<whatever> operating on
    # a list instead of a single filename, but is actually concurrently
    # controlled if there are multiple processors available.
    @needs_pr_init
    def calculate_score(self, contents, scorefxn_hash, params=None):
        """Calculates the score of a protein based on the provided contents of
        its PDB file. scorefxn_hash is not used inside the calculation, but is
        used for indexing into the buffer.
        """
        global PYROSETTA_ENV
        pose = self.get_pdb_contents_pose(contents, params)
        return PYROSETTA_ENV.scorefxn(pose)
    @needs_pr_init
    def calculate_rmsd(self, lhs_contents, rhs_path, rmsd_type,
                                      params=None):
        """Calculates the RMSD of two proteins from each other and stores
        it, without assuming commutativity."""
        pose_lhs = self.get_pdb_contents_pose(lhs_contents, params=params)
        pose_rhs = self.get_pdb_contents_pose(self.get_pdb_file_info(rhs_path) \
                                                  .get_contents(),
                                              params=params)
        return getattr(pr.rosetta.core.scoring, rmsd_type)(pose_lhs, pose_rhs)
    @needs_pr_init
    def calculate_neighbors(self, contents, coarsep=False, bound=None,
                            params=None):
        """Calculates the residue neighborhood matrix of a protein based on the
        provided contents of its PDB file."""
        pose = self.get_pdb_contents_pose(contents, params=params)
        result = []
        n_residues = pose.size()
        for i in range(1,n_residues+1):
            result.append([])
            for j in range(1,n_residues+1):
                if i > j:
                    result[-1].append(result[j-1][i-1])
                else:
                    result[-1].append(
                        # int for space efficiency in cache file
                        int(mpre.res_neighbors_p(pose,
                                                 i, j,
                                                 coarsep=coarsep,
                                                 bound=bound)))
        return result

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
        """Call a shell command:  shell echo 'Hello'  |  !echo 'Hello'"""
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
      analysis and plotting commands do nothing."""
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
        if self.timelimit == 0:
            print('No timelimit')
        else:
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
    def do_subplot(self, arg):
        """Create a subplot with Matlab syntax:  subplot 2 1 1"""
        print('\n\nsetting subplot '+arg+'\n\n')
        args = arg.split()
        if len(args) == 3:
            plt.subplot(*[int(a) for a in args])
        elif len(args) == 1:
            plt.subplot(int(args[0]))
    def do_save_plot(self, arg):
        """Save the plot currently in the plot buffer:  save_plot name.eps"""
        if MPIRANK == 0:
            if arg:
                plt.tight_layout()
                plt.savefig(arg, format=arg.split('.')[-1].lower())
            else:
                print('Your plot needs a name.')
    @continuous
    def do_plot_dir_rmsd_vs_score(self, arg):
        """For each PDB in a directory, plot the RMSDs vs a particular file
against their energy score, optionally specifying an upper bound on score:
    plot_dir_rmsd_vs_score indir infile.pdb  |
    plot_dir_rmsd_vs_score indir infile.pdb --params ABC  |
    plot_dir_rmsd_vs_score indir infile.pdb 4.6"""
        global PYROSETTA_ENV
        parser = argparse.ArgumentParser()
        parser.add_argument('in_dir',
                            action='store')
        parser.add_argument('in_file',
                            action='store')
        parser.add_argument('bound',
                            type=float,
                            action='store',
                            nargs='?',
                            default=0)
        parser.add_argument('--style',
                            dest='style',
                            action='store',
                            default='ro')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs='*',
                            default=None)
        parsed_args = parser.parse_args(arg.split())
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        params = None
        if parsed_args.params:
            params = []
            for param in parsed_args.params:
                if param.endswith('.params'):
                    params.append(param)
                else:
                    params.append(param+'.params')
        data = zip(self.data_buffer.gather_rmsd(
                       (os.path.join(parsed_args.in_dir, filename) \
                        for filename in filenames_in_in_dir \
                        if filename.endswith('.pdb')),
                       parsed_args.in_file,
                       'all_atom_rmsd',
                       params=params),
                   self.data_buffer.gather_score(
                       (os.path.join(parsed_args.in_dir, filename) \
                        for filename in filenames_in_in_dir \
                        if filename.endswith('.pdb')),
                       PYROSETTA_ENV.get_scorefxn_hash(),
                       params=params))
        rmsds,scores = zip(*(datapoint for datapoint in data if datapoint[1] < 0))
        plt.plot(rmsds, scores, parsed_args.style)
    @continuous
    def do_plot_dir_neighbors(self, arg):
        print('\n\n\n\ngonna plot neighbors for '+arg+'\n\n\n\n', flush=True)
        sys.stdout.flush()
        parser = argparse.ArgumentParser()
        parser.add_argument('in_dir',
                            action='store')
        parser.add_argument('start_i',
                            type=int,
                            action='store')
        parser.add_argument('end_i',
                            type=int,
                            action='store')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs='*',
                            default=None)
        parsed_args = parser.parse_args(arg.split())
        params = None
        if parsed_args.params:
            params = []
            for param in parsed_args.params:
                if param.endswith('.params'):
                    params.append(param)
                else:
                    params.append(param+'.params')
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        print('\n\nabout to find neighbors for '+arg+'\n\n', flush=True)
        matrices = self.data_buffer.gather_neighbors(
                       (os.path.join(parsed_args.in_dir, filename) \
                        for filename in filenames_in_in_dir \
                        if filename.endswith('.pdb')),
                       params=params)
        def m_valid_p(m, minsize):
            try:
                return len(m) >= minsize and \
                       functools.reduce(operator.and_, [len(v) >= minsize \
                                                        for v in m])
            except TypeError:
                return False
        matrices = [m for m in matrices if m_valid_p(m, parsed_args.end_i)]
        n_matrices = float(len(matrices))
        avg_matrix = []
        for x in range(parsed_args.start_i-1,parsed_args.end_i):
            avg_matrix.append([])
            for y in range(parsed_args.start_i-1,parsed_args.end_i):
                avg_matrix[-1].append(
                    functools.reduce(
                        operator.add,
                        [m[x][y] for m in matrices])/n_matrices)
        plt.imshow(avg_matrix, cmap='hot', interpolation='nearest',
                   extent=[parsed_args.start_i, parsed_args.end_i,
                           parsed_args.end_i, parsed_args.start_i],
                   aspect=1)
        print('\n\n\n\nfinished plotting neighbors for '+arg+'\n\n\n\n', flush=True)
        sys.stdout.flush()

if __name__ == "__main__":
    print('HELLO BEFORE CREATING COMMAND LINE', flush=True)
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
    print('HELLO AFTER CREATING COMMAND LINE', flush=True)
    if MPIRANK != 0:
        DEVNULL = open(os.devnull, 'w')
        sys.stdout = DEVNULL
    if parsed_args.script is not None:
        print('HELLO AFTER ENTERING LOOP', flush=True)
        with open(parsed_args.script) as f:
            OURCMDLINE.cmdqueue.extend(f.read().splitlines())
    if parsed_args.backgroundp:
        OURCMDLINE.cmdqueue.extend(['quit'])
    OURCMDLINE.cmdloop()
    if MPIRANK != 0:
        sys.stdout = STDOUT
        DEVNULL.close()
