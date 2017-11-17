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
import types, copy
import itertools, functools, operator
import csv
import hashlib
import os, io, time, argparse
import subprocess
import json
import cmd, shlex
import sys, traceback, inspect
import ast
import fcntl
## Other stuff
import timeout_decorator
from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg') # otherwise lack of display breaks program
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyrosetta as pr
import mszpyrosettaextension as mpre

DEBUG = True

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

### Decorators

def needs_pr_init(f):
    '''Makes sure PyRosetta gets initialized before f is called.'''
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
    '''Makes sure the scorefxn exists before f is called. (In order for the
    scorefxn to exist, pr.init() needs to have been called, so @needs_pr_init
    is unnecessary in front of this.)'''
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        global PYROSETTA_ENV
        if PYROSETTA_ENV.scorefxn is None:
            print('Scorefxn not initialized yet. Initializing from defaults...')
            PYROSETTA_ENV.set_scorefxn()
        return f(*args, **kwargs)
    return decorated
def uses_pr_env(f):
    '''Sets an attribute on the function that tells PDBDataBuffer to include a
    hash of the PyRosetta env as a storage key for the output of this function.
    Useless for functions that aren't calculate_ methods in PDBDataBuffer.'''
    f.uses_pr_env_p = True
    return f
def times_out(f):
    '''Makes a function time out after it hits self.timelimit.'''
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
    '''Makes a function in OurCmdLine repeat forever when continuous mode is
    enabled, and decorates it with times_out.'''
    @functools.wraps(f)
    @times_out
    def decorated(self_, *args, **kwargs):
        if self_.settings['continuous_mode']:
            while True:
                f(self_, *args, **kwargs)
        else:
            return f(self_, *args, **kwargs)
    return decorated
def pure_plotting(f):
    '''Wraps "pure plotting" cmd commands that shouldn't be executed unless
    plotting is enabled.'''
    @functools.wraps(f)
    def decorated(self_, *args, **kwargs):
        if self_.settings['plotting']:
            return f(self_, *args, **kwargs)
        else:
            return
    return decorated

### Useful functions
# vanilla stuff
def DEBUG_OUT(*args, **kwargs):
    if DEBUG:
        print('DEBUG: ', end='')
        kwargs['flush'] = True
        print(*args, **kwargs)
def get_filenames_from_dir_with_extension(dir_path, extension,
                                          strip_extensions_p=False):
    '''Returns a list of files from a directory with the path stripped, and
    optionally the extension stripped as well.'''
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
def process_limits(limits):
    '''Processes a limits arg for the cmd object. Returns a list of
    [lower_limit, upper_limit]. Sadly still has to be wrapped in a try/catch on
    the outside.'''
    return [(None if val.lower() in ('same', 'unbound', 'none') \
             else float(val)) \
            for val in limits]
def process_params(params):
    '''Process params list so that .params is appended if a path doesn't end
    with it.'''
    if params:
        return [(param if param.endswith('.params') \
                 else param+'.params') \
                for param in params]
    else:
        return None
# Rosetta stuff
@needs_pr_init
def pose_from_contents(contents, params=None):
    '''Returns a Pose of a given pdb file contents string.'''
    pose = None
    if params:
        pose = mpre.pose_from_pdbstring_with_params(contents, params)
    else:
        pose = pr.Pose()
        pr.rosetta.core.import_pose.pose_from_pdbstring(pose, contents)
    return pose
@needs_pr_init
def pose_from_stream(stream, params=None):
    '''Returns a Pose of a given stream pointing at the contents of a
    PDB.'''
    # not gonna bother rewinding stream
    return pose_from_contents(stream.read(), params)
# functions to generate different types of accessors for PDBDataBuffer
def make_PDBDataBuffer_get(data_name):
    '''Create a get_ accessor function for a PDBDataBuffer data type.'''
    # Python duck typing philosophy says we shouldn't do any explicit
    # checking of the thing stored at the attribute, but in any case,
    # it should be a function that takes at least one PDB file path as
    # a positional argument, a params list as a keyword argument, and a
    # bunch of hashable arguments as the rest of its arguments.
    def get(self_, *args, params=None, **kwargs):
        argspec = inspect.getfullargspec(
                      getattr(self_, 'calculate_'+data_name))
        first_defaulted_index = #TODO
        argtypes = {}
        argtypesindices = []
        for i, arg in enumerate(argspec.args):
            if arg.endswith('stream'):
                argtypes[i] = 'stream'
            elif arg.endswith('path'):
                argtypes[i] = 'path'
            else:
                argtypes[i] = 'other'
            argtypesindices.append(i)
        for kwarg in argspec.kwonlyargs:
            if kwarg.endswith('stream'):
                argtypes[kwarg] = 'stream'
            elif kwarg.endswith('path'):
                argtypes[kwarg] = 'path'
            else:
                argtypes[kwarg] = 'other'
            argtypesindices.append(kwarg)
        calcargs_proto   = []
        calckwargs_proto = {}
        for index in argtypesindices:
            if isinstance(index, int):
                if 
            elif isinstance(index, str):
        self_.retrieve_data_from_cache(os.path.dirname(file_path))
        file_info = self_.get_pdb_file_info(file_path)
        file_data = self_.data.setdefault(file_info.hash, {}) \
                              .setdefault(data_name, {})
        kwargs_args_list = list(kwargs.keys())
        kwargs_args_list.sort()
        accessor_string = str(tuple(args) + \
                              tuple(kwargs[arg] for arg in kwargs_args_list))
        try:
            return file_data[accessor_string]
        except KeyError:
            # construct new args
            if self_.calculatingp:
                file_data[accessor_string] = \
                    getattr(self_, 'calculate_'+data_name) \
                        (*calcargs, params=params, **calckwargs)
                self_.changed_dirs.add(os.path.dirname(file_path))
                self_.update_caches()
                return file_data[accessor_string]
            else:
                raise KeyError('That file\'s not in the cache.')
    get.__name__ = 'get_'+data_name
    get.__doc__  = 'Call calculate_'+data_name+ \
                   '() on a file path with caching magic.'
    return get

def make_PDBDataBuffer_gather(data_name):
    '''Make a gather_ accessor for a PDBDataBuffer that concurrently operates
    on a list of filenames instead of a single filename.'''
    def gather(self_, file_list, *args, params=None, **kwargs):
        abspaths = tuple(os.path.abspath(path) for path in file_list)
        if MPISIZE == 1 or not self_.calculatingp:
            result = []
            for file_path in abspaths:
                try:
                    result.append(
                        getattr(self_, 'get_'+data_name) \
                                   (file_path, *args,
                                    params=params, **kwargs))
                except KeyError:
                    pass
            return result
        else:
            # getting this instead of a path makes a worker thread die:
            QUIT_GATHERING_SIGNAL = -1
            READY_TAG = 0 # used once by worker, to sync startup
            DONE_TAG  = 1 # used by worker to pass result and request new job
            WORK_TAG  = 2 # used by master to pass next path to worker
            # will be used to index into self_.data to retrieve final result:
            file_indices_dict = {}
            DEBUG_OUT('retrieving caches for gather_ operation')
            if MPIRANK == 0:
                current_file_index = 0
                working_threads = MPISIZE-1
                def save_result(result):
                    file_info_dict, file_data, accessor_string = result
                    DEBUG_OUT('about to save result for '+file_info_dict['path'])
                    self_.pdb_paths[file_info_dict['path']] = \
                        {'mtime': file_info_dict['mtime'],
                         'hash':  file_info_dict['hash']}
                    self_.data.setdefault(file_info_dict['hash'], {}) \
                              .setdefault(data_name, {}) \
                              [accessor_string] = file_data
                    file_indices_dict[file_info_dict['path']] = \
                        (file_info_dict['hash'], data_name, accessor_string)
                    DEBUG_OUT('just saved result for '+file_info_dict['path'])
                for dirname in set((os.path.dirname(path) \
                                    for path in abspaths)):
                    DEBUG_OUT('retrieving caches for '+dirname)
                    self_.retrieve_data_from_cache(dirname)
                DEBUG_OUT('all caches retrieved')
                while True:
                    result = MPICOMM.recv(source=MPI.ANY_SOURCE,
                                          tag=MPI.ANY_TAG,
                                          status=MPISTATUS)
                    result_source = MPISTATUS.Get_source()
                    result_tag    = MPISTATUS.Get_tag()
                    DEBUG_OUT('data received from '+str(result_source))
                    # try to find a PDB we don't have the data for yet:
                    while current_file_index < len(abspaths):
                        file_path = abspaths[current_file_index]
                        file_info = self_.get_pdb_file_info(file_path)
                        file_data = self_.data.setdefault(file_info.hash, {}) \
                                              .setdefault(data_name, {})
                        kwargs_args_list = list(kwargs.keys())
                        kwargs_args_list.sort()
                        accessor_string = str(tuple(args) + \
                                              tuple(kwargs[arg] \
                                                    for arg \
                                                    in kwargs_args_list))
                        file_info_dict = {'path':  file_info.path,
                                          'mtime': file_info.mtime,
                                          'hash':  file_info.hash}
                        try:
                            save_result([file_info_dict,
                                         file_data[accessor_string],
                                         accessor_string])
                            DEBUG_OUT('increasing current_file_index from %d to %d' % \
                                      (current_file_index, current_file_index+1))
                            current_file_index += 1
                        except KeyError:
                            break
                    if current_file_index < len(abspaths):
                        DEBUG_OUT('assigning ' + file_info.path+' to ' + \
                                  str(result_source))
                        MPICOMM.send([file_info_dict,
                                      file_path,
                                      accessor_string],
                                     dest=result_source, tag=WORK_TAG)
                        DEBUG_OUT('assignment sent')
                        current_file_index += 1
                    else:
                        DEBUG_OUT('all files done. killing '+str(result_source))
                        MPICOMM.send(QUIT_GATHERING_SIGNAL,
                                     dest=result_source, tag=WORK_TAG)
                        DEBUG_OUT('worker killed')
                        working_threads -= 1
                    if result_tag == DONE_TAG:
                        save_result(result)
                        self_.changed_dirs.add(
                            os.path.dirname(
                                result[0]['path']))
                        self_.update_caches()
                    if working_threads <= 0:
                        break
            else:
                MPICOMM.send(None, dest=0, tag=READY_TAG)
                while True:
                    package = MPICOMM.recv(source=0, tag=WORK_TAG)
                    if package == QUIT_GATHERING_SIGNAL:
                        break
                    file_info_dict, file_path, accessor_string = package
                    contents = open(file_path, 'r').read()
                    result_data = \
                        getattr(self_, 'calculate_'+data_name) \
                            (contents, *args,
                             params=params, **kwargs)
                    MPICOMM.send(copy.deepcopy([file_info_dict,
                                                result_data,
                                                accessor_string]),
                                 dest=0, tag=DONE_TAG)
            # Synchronize everything that could have possibly changed:
            self_.data = MPICOMM.bcast(self_.data, root=0)
            self_.file_paths = MPICOMM.bcast(self_.pdb_paths, root=0)
            self_.cache_paths = MPICOMM.bcast(self_.cache_paths, root=0)
            file_indices_dict = MPICOMM.bcast(file_indices_dict, root=0)
            # All threads should be on the same page at this point.
            retlist = []
            for path in abspaths:
                indices = file_indices_dict[path]
                retlist.append(self_.data[indices[0]][indices[1]][indices[2]])
            return retlist
    gather.__name__ = 'gather_'+data_name
    gather.__doc__  = 'Call calculate_'+data_name+ \
                      '() on a list of file paths with concurrency magic.'
    return gather

### Housekeeping classes

class PyRosettaEnv():
    '''Stores stuff relating to PyRosetta, like whether pr.init() has been
    called.'''
    def __init__(self):
        self.initp = False
        self.scorefxn = None
    @needs_pr_init
    def set_scorefxn(self, name=None, patch=None):
        '''Sets the scorefxn and optionally applies a patch. Defaults to
        get_fa_scorefxn() with no arguments.'''
        if name is None:
            self.scorefxn = pr.get_fa_scorefxn()
        else:
            self.scorefxn = pr.create_score_function(name)
        if patch:
            self.scorefxn.apply_patch_from_file(patch)
    @needs_pr_scorefxn
    def get_hash(self):
        '''Gets a hash of the properties of the current env.'''
        hash_fun = hashlib.md5()
        weights = self.scorefxn.weights()
        hash_fun.update(weights.weighted_string_of(weights).encode())
        return hash_fun.hexdigest()

class PDBDataBuffer():
    '''Singleton class that stores information about PDBs, to be used as this
    program's data buffer. Its purpose is to intelligently abstract the
    retrieval of information about PDBs in such a way that the information is
    cached and/or buffered in the process. The only methods in it that should
    ideally be used externally are the data retrieval methods, and out of those
    ideally just the caching ones (currently get_ and gather_ methods). In
    practice, update_caches() is also used externally to force a cache update.

    Internally, it holds a data dict, which indexes all the data first by the
    MD5 hash of the contents of the PDB that produced the data, then the type
    of data it is (e.g. RMSD or score), and finally the parameters that
    produced the data (like the particular weights on the scorefunction for a
    score). It also holds a dict of paths to PDBs, the contents hashes of these
    PDBs, and the mtime of the PDB at which the contents hash was produced, so
    that if it is asked about a PDB whose hash is already known, it does not
    need to recalculate it. Finally, it also holds a dict mapping the caches
    it's already loaded to their mtimes at which they were loaded, so that it
    does not load a cache unless it's changed.

    On disk, the data is cached in a .paperclip_cache file that contains a JSON
    array of the data dict and the PDB paths dict, each with only the entries
    for the PDB files in the same folder as the cache. If caching is turned on,
    whenever the buffer generates data for a file, it will check whether the
    file's folder has a cache with that data yet, and if not, it will write one
    (making sure to load any cached data in that folder that already exists).
    Caching is done in a thread-safe manner, with file locks. Throughout the
    reading, updating, and writing of a cache, the buffer will maintain an
    exclusive lock on the cache, so that it doesn't get updated on disk in
    between its reading and updating/writing, which would cause the first
    update to get lost.

    The ability to get a particular type of data is added to the buffer by
    defining a calculate_ method. Corresponding get_ and gather_ methods are
    created dynamically at initialization, which add caching to the calculate_
    method, and take a filename or list of filenames instead of a file contents
    string. For this reason, all calculate_ methods should take a contents
    string as their first argument, and also a params keyword argument for the
    file params.

    The internal settings calculatingp and plottingp work like this:

      - calculatingp = False: The buffer will only retrieve cached information,
          never calculating its own data or saving caches. get_ operations will
          return a KeyError if the data for a PDB is not in the caches it's
          loaded, and gather_ operations will leave a PDB's data out of the
          list they return if they can't find it.
      - calculatingp = True, cachingp = False: The buffer will calculate new
          information if it doesn't have it, but never save it. Useful if
          real-time caching is taking too much time, and if instead you want to
          do it manually by calling update_caches(force=True) at select times.
          Note that with cachingp = False, update_caches() doesn't do anything
          unless you call it with force=True. (This is the case with
          calculatingp = False as well, making it the only real effect of
          cachingp in that case.)
      - calculatingp = True, cachingp = True: The buffer will calculate new
          information if it doesn't have it, and save it to disk immediately
          all of the time.

    The structure of the buffer's data variables, although summarized above,
    can be stated more succinctly as:

    self.data:
    { content_key : { data_function_name :
                        { data_function_params_tuple : data } } }
    self.file_paths:
    { file_path : { 'mtime' : mtime_at_last_hashing,
                    'hash'  : contents_hash } }
    self.cache_paths:
    { cache_file_dir_path : mtime_at_last_access }
    '''
    ## Core functionality
    def __init__(self):
        self.calculatingp = True
        self.cachingp = MPIRANK == 0
        self.data = {}
        self.file_paths = {}
        self.cache_paths = {}
        self.changed_dirs = set()
        # Monkey patch in the data accessors:
        attribute_names = dir(self)
        for data_name in (attribute_name[10:] \
                          for attribute_name in attribute_names \
                          if attribute_name.startswith('calculate_')):
            setattr(self, 'get_'+data_name,
                    types.MethodType(make_PDBDataBuffer_get(data_name), self))
            setattr(self, 'gather_'+data_name,
                    types.MethodType(make_PDBDataBuffer_gather(data_name),
                                     self))
    def retrieve_data_from_cache(self, dirpath, cache_fd=None):
        '''Retrieves data from a cache file of a directory. The data in the
        file is a JSON of a data dict of a PDBDataBuffer, except instead of
        file paths, it has just filenames.'''
        if MPIRANK != 0:
            return
        absdirpath = os.path.abspath(dirpath)
        try:
            cache_path = os.path.join(absdirpath, '.paperclip_cache')
            diskmtime = os.path.getmtime(cache_path)
            ourmtime = self.cache_paths.setdefault(absdirpath, 0)
            if diskmtime > ourmtime:
                file_contents = None
                retrieved = None
                if cache_fd is not None:
                    pos = cache_fd.tell()
                    cache_fd.seek(0)
                    file_contents = cache_fd.read()
                    DEBUG_OUT('cache at '+cache_fd.name+' read')
                    cache_fd.seek(pos)
                else:
                    try:
                        #DEBUG_OUT('opening and retrieving cache at'+cache_path)
                        with open(cache_path, 'r') as cache_file:
                            file_contents = cache_file.read()
                            DEBUG_OUT('cache at '+cache_file.name+' read')
                    except FileNotFoundError:
                        pass
                try:
                    retrieved = json.loads(file_contents)
                except json.decoder.JSONDecodeError:
                    try:
                        retrieved = ast.literal_eval(file_contents)
                    except SyntaxError:
                        pass
                if retrieved is not None:
                    DEBUG_OUT('cache retrieved')
                    diskdata, disk_pdb_info = retrieved
                    for content_key, content in diskdata.items():
                        #DEBUG_OUT('reading data for content key '+content_key)
                        for data_name, data_keys in content.items():
                            #DEBUG_OUT('  reading data type '+data_name)
                            for data_key, data in data_keys.items():
                                #DEBUG_OUT('    reading data key '+str(data_key))
                                self.data.setdefault(content_key,{})
                                self.data[content_key].setdefault(data_name,{})
                                # str(data_key) for backwards compatibility
                                self.data[content_key] \
                                         [data_name] \
                                         [str(data_key)] = data
                    for name, info_pair in disk_pdb_info.items():
                        DEBUG_OUT('saving info for '+name)
                        path = os.path.join(absdirpath, name)
                        our_info_pair = self.file_paths.get(pdb_path, None)
                        ourfilemtime = None
                        if our_info_pair is not None:
                            ourfilemtime = our_info_pair['mtime']
                        else:
                            ourfilemtime = 0
                        if info_pair['mtime'] > ourfilemtime:
                            self.file_paths[path] = info_pair
            self.cache_paths[absdirpath] = diskmtime
        except FileNotFoundError:
            pass
    def update_caches(self, force=False):
        '''Updates the caches for every directory the cache knows about. This
        method is thread-safe.'''
        if (not (self.cachingp or force)) or MPIRANK != 0:
            return
        dir_paths = {}
        for path in self.file_paths.keys():
            dir_path, name = os.path.split(path)
            if dir_path in self.changed_dirs:
                dir_paths.setdefault(dir_path, set())
                dir_paths[dir_path].add(name)
        for dir_path, names in dir_paths.items():
            cache_path = os.path.join(dir_path, '.paperclip_cache')
            cache_file = None
            try:
                cache_file = open(cache_path, 'r+')
            except FileNotFoundError:
                cache_file = open(cache_path, 'w+')
            while True:
                try:
                    fcntl.flock(cache_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            self.retrieve_data_from_cache(dir_path, cache_fd=cache_file)
            dir_data = {}
            dir_file_paths = {}
            for name in names:
                path = os.path.join(dir_path, name)
                dir_file_paths[name] = self.file_paths[path]
                ourhash = dir_file_paths[name]['hash']
                if name.endswith('.pdb'):
                    try:
                        dir_data[ourhash] = self.data[ourhash]
                    except KeyError:
                        pass
                #DEBUG_OUT('  updated cache with data for file at ', pdb_path)
            cache_file.write(json.dumps([dir_data, dir_file_paths]))
            fcntl.flock(cache_file, fcntl.LOCK_UN)
            cache_file.close()
            DEBUG_OUT('wrote cache file at ', cache_path)
            self.cache_paths[dir_path] = os.path.getmtime(cache_path)
            self.changed_dirs = set()
    def encapsulate_file(self, path):
        '''Returns an object that in theory contains the absolute path, mtime, contents
        hash, and maybe contents of a file. The first three are accessed
        directly, while the last is accessed via an accessor method, so that it
        can be retrieved if necessary. Creating and updating the object both
        update the external PDBDataBuffer's info on that pdb.'''
        class FileInfo():
            def __init__(self_, path_):
                self_.path = os.path.abspath(path_)
                self_.file_paths_dict = \
                    self.file_paths.setdefault(self_.path, {})
                self_.mtime = self_.file_paths_dict.setdefault('mtime', 0)
                self_.hash = self_.file_paths_dict.setdefault('hash', None)
                self_.stream = None
                diskmtime = os.path.getmtime(self_.path)
                if diskmtime > self_.mtime or self_.hash is None:
                    self_.update(diskmtime)
            def update(self_, mtime = None):
                diskmtime = mtime or os.path.getmtime(self_.path)
                if diskmtime > self_.mtime or self_.contents is None:
                    with open(self_.path, 'r') as pdb_file:
                        contents = pdb_file.read()
                    self_.mtime = diskmtime
                    hash_fun = hashlib.md5()
                    hash_fun.update(self_.contents.encode())
                    self_.hash = hash_fun.hexdigest()
                    self_.stream = io.StringIO(contents)
                    self_.file_paths_dict['mtime'] = self_.mtime
                    self_.file_paths_dict['hash'] = self_.hash
            def get_stream(self_, mtime = None):
                self_.update(mtime)
                return self_.stream
        return FileInfo(path)

    ## Auxiliary stuff

    ## Calculating Rosetta stuff
    # Each calculate_<whatever> also implicity creates a get_<whatever>, which
    # is just the same thing but with all the buffer/caching magic attached, 
    # and with stream args replaced with path args. It also creates a
    # gather_<whatever>, which outwardly looks like get_<whatever> operating on
    # a list instead of a single filename, but is actually concurrently
    # controlled if there are multiple processors available.
    @uses_pr_env
    @needs_pr_init
    def calculate_score(self, stream, params=None):
        '''Calculates the score of a protein from a stream of its PDB file.
        scorefxn_hash is not used inside the calculation, but is used for
        indexing into the buffer.
        '''
        global PYROSETTA_ENV
        pose = pose_from_stream(stream, params)
        return PYROSETTA_ENV.scorefxn(pose)
    @needs_pr_init
    def calculate_rmsd(self, lhs_stream, rhs_stream, rmsd_type,
                                      params=None):
        '''Calculates the RMSD of two proteins from each other and stores
        it, without assuming commutativity.'''
        pose_lhs = pose_from_stream(lhs_stream, params=params)
        pose_rhs = pose_from_stream(rhs_stream, params=params)
        return getattr(pr.rosetta.core.scoring, rmsd_type)(pose_lhs, pose_rhs)
    @needs_pr_init
    def calculate_neighbors(self, stream, coarsep=False, bound=None,
                            params=None):
        '''Calculates the residue neighborhood matrix of a protein based on the
        provided contents of its PDB file.'''
        pose = pose_from_stream(stream, params=params)
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
    '''Singleton class for our interactive CLI.'''
    ## Built-in vars
    intro = 'Welcome to PAPERCLIP. Type help or ? to list commands.'
    prompt = '* '
    ## Our vars (don't wanna mess with __init__)
    cmdfile = None
    settings = {'calculation': True,
                'caching': True,
                'plotting': MPIRANK == 0,
                'continuous_mode': False}
    timelimit = 0
    last_im = None
    ## The two buffers:
    data_buffer = PDBDataBuffer() # contains computed data about PDBs
    text_buffer = io.StringIO()   # contains text output, formatted as a TSV
    # There is also a plot buffer, but that is contained within pyplot.

    ## Housekeeping
    def do_quit(self, arg):
        '''Stop recording and exit:  quit'''
        self.close()
        return True
    def do_bye(self, arg):
        '''Stop recording and exit:  bye'''
        return self.do_quit(arg)
    def do_EOF(self, arg):
        '''Stop recording and exit:  EOF  |  ^D'''
        return self.do_quit(arg)
    def do_exit(self, arg):
        '''Stop recording and exit:  exit'''
        return self.do_quit(arg)
    def emptyline(self):
        pass

    ## Making argparse-generated docs work
    def preloop(self):
        # Most of the following was copied from the original cmd.py with very
        # little modification; I hereby absolve myself of any responsibility
        # regarding it
        names = self.get_names()
        cmds_undoc = []
        help = set()
        for name in names:
            if name[:5] == 'help_':
                help.add(name[5:])
        names.sort()
        # There can be duplicates if routines overridden
        prevname = ''
        for name in names:
            if name[:3] == 'do_':
                if name == prevname:
                    continue
                prevname = name
                cmd=name[3:]
                if not ((cmd in help) or (getattr(self, name).__doc__)):
                    cmds_undoc.append(cmd)
        # This is my code. It runs the undocumented flags with the '-h' flag
        # and output silenced, which allows the command's __doc__ to be
        # overwritten but not the rest of the script to be executed (since in
        # that case, the parse_args() line throws a SystemExit exception, which
        # I manually catch and turn into a return).
        with open(os.devnull, 'w') as DEVNULL:
            sys.stdout = DEVNULL
            for cmd in cmds_undoc:
                self.onecmd(cmd+' -h')
        sys.stdout = STDOUT

    ## Parsing
    def get_arg_position(self, text, line):
        '''For completion; gets index of current positional argument (returns 1 for
        first arg, 2 for second arg, etc.).'''
        return len(line.split()) - (text != '')

    ## Recording and playing back commands
    def do_record(self, arg):
        '''Save future commands to filename:  record plot.cmd'''
        self.cmdfile = open(arg, 'w')
    def do_playback(self, arg):
        '''Play back commands from a file:  playback plot.cmd'''
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
        '''Call a shell command:  shell echo 'Hello'  |  !echo 'Hello' '''
        os.system(arg)
    def do_cd(self, arg):
        '''Change the current working directory:  cd dir'''
        try:
            os.chdir(arg)
        except FileNotFoundError:
            print('No such file or directory: ' + arg)
    def do_mv(self, arg):
        '''Call the shell command mv:  mv a b'''
        self.do_shell('mv ' + arg)
    def do_rm(self, arg):
        '''Call the shell command rm:  rm a'''
        self.do_shell('rm ' + arg)
    def do_ls(self, arg):
        '''Call the shell command ls:  ls ..'''
        self.do_shell('ls ' + arg)
    def do_pwd(self, arg):
        '''Get the current working directory:  pwd'''
        os.getcwd()

    ## Settings
    def do_get_settings(self, arg):
        '''Print settings of current session:  get_settings'''
        for key, value in self.settings.items():
            transformed_value = 'yes' if value == True else \
                                'no'  if value == False else value
            print('{0:<20}{1:>8}'.format(key+':', transformed_value))
    def do_set(self, arg):
        '''Set or toggle a yes/no setting variable in the current session.
    set calculation no  |  set calculation

Available settings are:
  caching: Whether to cache the results of calculations or not.
  calculation: Whether to perform new calculations for values that may be
      outdated in the cache, or just use the possibly outdated cached values.
  continuous_mode: Repeat all analysis and plotting commands until they hit the
      time limit, or forever. Useful if a program is still generating data for
      a directory, but you want to start caching now. (To set a time limit, use
      the command 'set_timelimit'.)
  plotting: Whether to actually output plots, or to just perform and cache the
      calculations for them. Disabling both this and 'calculation' makes most
      analysis and plotting commands do nothing.'''
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
            if str.lower(varvalue) in ('yes','y','t','true'):
                value = True
            elif str.lower(varvalue) in ('no','n','f','false'):
                value = False
            else:
                print('That\'s not a valid setting value. Try \'yes\' or'
                      '\'no\'.')
                return
            if varname in self.settings.keys():
                if varname == 'plotting':
                    self.settings['plotting'] = MPIRANK == 0 and value
                else:
                    self.settings[varname] = value
            else:
                print('That\'s not a valid setting name. Try '
                      '\'get_settings\'.')
                return
        self.data_buffer.calculatingp = self.settings['calculation']
        self.data_buffer.cachingp     = self.settings['caching']
    def complete_set(self, text, line, begidx, endidx):
        position = self.get_arg_position(text, line)
        if position == 1:
            return [i for i in list(self.settings.keys()) if i.startswith(text)]
        elif position == 2:
            return [i for i in ['yes', 'no'] if i.startswith(text)]
    def do_get_timelimit(self, arg):
        '''Print the current time limit set on analysis commands.
    get_timelimit'''
        if self.timelimit == 0:
            print('No timelimit')
        else:
            print(str(self.timelimit)+' seconds')
    def do_set_timelimit(self, arg):
        '''Set a time limit on analysis commands, in seconds. Leave as 0 to let
commands run indefinitely.
    set_timelimit 600'''
        try:
            self.timelimit = int(arg)
        except ValueError:
            print('Enter an integer value of seconds.')

    ## Buffer interaction
    # Data buffer
    def do_clear_data(self, arg):
        '''Clear the data buffer of any data:  clear_data'''
        calculatingp = self.data_buffer.calculatingp
        cachingp     = self.data_buffer.cachingp
        self.data_buffer = PDBDataBuffer()
        self.data_buffer.calculatingp = calculatingp
        self.data_buffer.cachingp     = cachingp
    def do_update_caches(self, arg):
        '''Update the caches for the data buffer:  update_caches'''
        self.data_buffer.update_caches(force=True)
    # Text buffer
    def do_clear_text(self, arg):
        '''Clear the text buffer of any text output:  clear_text'''
        self.text_buffer.close()
        self.text_buffer = io.StringIO()
    def do_view_text(self, arg):
        '''View the text buffer, less-style:  view_text'''
        subprocess.run(['less'],
                       input=bytes(self.text_buffer.getvalue(), 'utf-8'))
    def do_save_text(self, arg):
        parser = argparse.ArgumentParser(
            description='Save the text buffer to a file, optionally specifying'
                        ' a different format.')
        parser.add_argument('path',
                            help='Output path.')
        parser.add_argument('--format',
                            dest='format',
                            nargs='?',
                            default=None,
                            help='Format to save text buffer in.')
        self.do_save_text.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(arg.split())
        except SystemExit:
            return
        out_format = None
        if parsed_args.format is None:
            last_segment = parsed_args.path.split('.')[-1]
            if last_segment.lower() in ('tsv','csv'):
                out_format = last_segment.lower()
            else:
                out_format = 'tsv'
        else:
            out_format = parsed_args.format.lower()
        if out_format == 'tsv':
            try:
                open(parsed_args.path, 'w').write(self.text_buffer.getvalue())
            except FileNotFoundError:
                print('Invalid output path.')
        elif out_format == 'csv':
            reader = csv.reader(self.text_buffer, delimiter='\t')
            try:
                with open(parsed_args.path, 'w') as out_file:
                    out_file = csv.writer(out_file)
                    self.text_buffer.seek(0)
                    for row in reader:
                        out_file.writerow(row)
            except FileNotFoundError:
                print('Invalid output path.')
        else:
            print('Invalid output format.')
            
    # Plot buffer
    def do_clear_plot(self, arg):
        '''Clear the plot buffer:  clear_plot'''
        self.last_im = None
        plt.cla()
        plt.clf()

    ## Basic Rosetta stuff
    def do_get_scorefxn(self, arg):
        '''Print the name of the current scorefxn, if any:  get_scorefxn'''
        global PYROSETTA_ENV
        if PYROSETTA_ENV.scorefxn is None:
            print('No scorefxn currently set.')
        else:
            print(PYROSETTA_ENV.scorefxn.get_name())
    def do_set_scorefxn(self, arg):
        '''Set the current scorefxn, optionally applying a patchfile:
    set_scorefxn ref2015  |  set_scorefxn ref2015 docking'''
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

    ## Matplotlib stuff
    # fundamental stuff
    @pure_plotting
    def do_save_plot(self, arg):
        '''Save the plot currently in the plot buffer:  save_plot name.eps'''
        # plotting should already be off, but it doesn't hurt to double-check
        if MPIRANK == 0:
            if arg:
                try:
                    plt.savefig(arg, format=arg.split('.')[-1].lower())
                except:
                    print('Valid extensions are .png, .pdf, .ps, .eps, and '
                          '.svg.')
            else:
                print('Your plot needs a name.')
    @pure_plotting
    def do_plot_size(self, arg):
        '''Set the plot size in inches:  plot_size 10 10'''
        try:
            plt.gcf().set_size_inches(*(float(x) for x in arg.split()))
        except:
            print('Provide two numbers separated by spaces.')
    # titles and labels
    @pure_plotting
    def do_plot_title(self, arg):
        '''Set the title of the current plot:  plot_title My title'''
        plt.title(arg)
    @pure_plotting
    def do_plot_xlabel(self, arg):
        '''Set the x axis label of the current plot:  plot_xlabel My xlabel'''
        plt.xlabel(arg)
    @pure_plotting
    def do_plot_ylabel(self, arg):
        '''Set the y axis label of the current plot:  plot_ylabel My ylabel'''
        plt.ylabel(arg)
    @pure_plotting
    def do_xticks(self, arg):
        '''Set the xticks on your plot, optionally specifying location.
    xticks 'label 1' 'label 2' 'label 3' |
    xticks label\ 1  label\ 2  label\ 3  |
    xticks ('label 1', 0.1), ('label 2', 0.2), ('label 3', 0.3)'''
        tick_indices = []
        tick_labels  = []
        try:
            parsed = ast.literal_eval(arg)
            try:
                tick_labels, tick_indices = zip(parsed)
                tick_indices = np.array(tick_indices)
            except:
                print('Malformed input. See examples in help.')
        except SyntaxError:
            try:
                parsed = ast.literal_eval('\''+'\',\''.join(shlex.split(arg))+'\'')
                tick_indices = np.arange(len(parsed))
                tick_labels = parsed
            except:
                print('Malformed input. See examples in help.')
        plt.xticks(tick_indices, tick_labels)
    @pure_plotting
    def do_yticks(self, arg):
        '''Set the xticks on your plot, optionally specifying location.
    yticks 'label 1' 'label 2' 'label 3' |
    yticks label\ 1  label\ 2  label\ 3  |
    yticks ('label 1', 0.1), ('label 2', 0.2), ('label 3', 0.3)'''
        tick_indices = []
        tick_labels  = []
        try:
            parsed = ast.literal_eval(arg)
            try:
                tick_labels, tick_indices = zip(parsed)
                tick_indices = np.array(tick_indices)
            except:
                print('Malformed input. See examples in help.')
        except SyntaxError:
            try:
                parsed = ast.literal_eval('\''+'\',\''.join(shlex.split(arg))+'\'')
                tick_indices = np.arange(len(parsed))
                tick_labels = parsed
            except:
                print('Malformed input. See examples in help.')
        plt.yticks(tick_indices, tick_labels)
    @pure_plotting
    def do_set_xticks_rotation(self, arg):
        '''Set the rotation of the xtick labels in your plot.
    set_xticks_rotation 90'''
        try:
            if arg not in ('horizontal', 'vertical'):
                arg = float(arg)
        except:
            print('Invalid rotation value. It should be "horizontal", '
                  '"vertical", or a number.')
        plt.setp(plt.xticks()[1], rotation=arg)
    @pure_plotting
    def do_set_yticks_rotation(self, arg):
        '''Set the rotation of the ytick labels in your plot.
    set_yticks_rotation 90'''
        try:
            if arg not in ('horizontal', 'vertical'):
                arg = float(arg)
        except:
            print('Invalid rotation value. It should be "horizontal", '
                  '"vertical", or a number.')
        plt.setp(plt.yticks()[1], rotation=arg)
    @pure_plotting
    def do_prune_xticks(self, arg):
        '''Remove every other xtick:  prune_xticks'''
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[1:-1:2])
    # axes stuff
    def do_xlim(self, arg):
        '''Set limits for the x axis.
    xlim 0 1  |  xlim SAME 1'''
        try:
            left, right = arg.split()
            left  = None if left.lower()  == 'same' else float(left)
            right = None if right.lower() == 'same' else float(right)
            plt.gca().set_xlim(left=left, right=right)
        except ValueError:
            print('Specify a value for each side of the limits. If you want to'
                  'leave it the same, write "SAME".')
    @pure_plotting
    def do_ylim(self, arg):
        '''Set limits for the y axis.
    ylim 0 1  |  ylim SAME 1'''
        try:
            left, right = arg.split()
            left  = None if left.lower()  == 'same' else float(left)
            right = None if right.lower() == 'same' else float(right)
            plt.gca().set_ylim(left=left, right=right)
        except ValueError:
            print('Specify a value for each side of the limits. If you want to'
                  'leave it the same, write "SAME".')
    @pure_plotting
    def do_invert_xaxis(self, arg):
        '''Invert the current axes' x axis:  invert_xaxis'''
        plt.gca().invert_xaxis()
    @pure_plotting
    def do_invert_yaxis(self, arg):
        '''Invert the current axes' y axis:  invert_yaxis'''
        plt.gca().invert_yaxis()
    # subplot stuff
    @pure_plotting
    def do_subplot(self, arg):
        '''Create a subplot with Matlab syntax:  subplot 2 1 1'''
        try:
            args = arg.split()
            if len(args) == 3:
                plt.subplot(*[int(a) for a in args])
            elif len(args) == 1:
                plt.subplot(int(args[0]))
        except RuntimeError:
            print('That\'s not a valid subplot spec.')
    @pure_plotting
    def do_subplot2grid(self, arg):
        parser = argparse.ArgumentParser(
            description='Create a subplot grid, first specifying the grid '
                        'size, and then the subplot location on the grid. '
                        'Optionally include a colspan or rowspan.')
        parser.add_argument('specs',
                            nargs=4,
                            type=int,
                            help='Four integers: the grid height and width, '
                                 'then the coordinates of the location you\'re'
                                 ' selecting (indexed from 0).')
        parser.add_argument('--colspan',
                            dest='colspan',
                            action='store',
                            type=int,
                            default=1,
                            help='Amount of columns your selection spans for.')
        parser.add_argument('--rowspan',
                            dest='rowspan',
                            action='store',
                            type=int,
                            default=1,
                            help='Amount of rows your selection spans for.')
        self.do_subplot2grid.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(arg.split())
        except SystemExit:
            return
        plt.subplot2grid(tuple(parsed_args.specs[0:2]),
                         tuple(parsed_args.specs[2:4]),
                         colspan=parsed_args.colspan,
                         rowspan=parsed_args.rowspan)
    @pure_plotting
    def do_tight_layout(self, arg):
        '''Adjust subplot spacing so that there's no overlaps between
different subplots.
    tight_layout'''
        plt.tight_layout()
    @pure_plotting
    def do_add_colorbar(self, arg):
        '''Add a colorbar to your figure next to a group of subplots, for the
most recently plotted subplot. Don't call tight_layout after this; that breaks
everything.
    add_colorbar'''
        SPACE   = 0.2
        PADDING = 0.7
        plt.tight_layout()
        fig = plt.gcf()
        w,h = fig.get_size_inches()
        fig.set_size_inches((w/(1-SPACE), h))
        fig.subplots_adjust(right=1-SPACE)
        cbax = fig.add_axes([1-SPACE*(1-PADDING/2), 0.15,
                             SPACE*(1-PADDING),     0.7])
        fig.colorbar(self.last_im, cax=cbax)
    ## Calculations stuff
    # Text
    @continuous
    def do_table_dir_rmsd_vs_score(self, arg):
        global PYROSETTA_ENV
        parser = argparse.ArgumentParser(
            description='Add a table of RMSDs vs a particular file and scores '
                        'for a directory of PDBs to the text buffer.')
        parser.add_argument('in_dir',
                            action='store',
                            help='The directory to plot for.')
        parser.add_argument('in_file',
                            action='store',
                            help='The PDB to compare with for RMSD.')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs='*',
                            default=None,
                            help='Params files for the PDBs.')
        parser.add_argument('--rmsdlim',
                            nargs='*',
                            help='Limits on RMSD for the plot. If a side\'s '
                                 'limit is given as NONE, that side is '
                                 'unlimited.')
        parser.add_argument('--scorelim',
                            nargs='*',
                            help='Limits on score for the plot. If a side\'s '
                                 'limit is given as NONE, that side is '
                                 'unlimited.')
        parser.add_argument('--sorting',
                            dest='sorting',
                            action='store',
                            default='scoreinc',
                            help='Sorting criterion. Can be rmsddec, rmsdinc, '
                                 'scoredec, or scoreinc.')
        self.do_table_dir_rmsd_vs_score.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(arg.split())
        except SystemExit:
            return
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        filenames_in_in_dir = [filename \
                               for filename in filenames_in_in_dir \
                               if filename.endswith('.pdb')]
        params = process_params(parsed_args.params)
        data = zip(filenames_in_in_dir,
                   self.data_buffer.gather_rmsd(
                       (os.path.join(parsed_args.in_dir, filename) \
                        for filename in filenames_in_in_dir),
                       parsed_args.in_file,
                       'all_atom_rmsd',
                       params=params),
                   self.data_buffer.gather_score(
                       (os.path.join(parsed_args.in_dir, filename) \
                        for filename in filenames_in_in_dir),
                       params=params))
        # sorting criterion
        criterion = None
        if parsed_args.sorting.lower() == 'rmsddec':
            criterion = lambda p: -p[1]
        elif parsed_args.sorting.lower() == 'rmsdinc':
            criterion = lambda p: p[1]
        elif parsed_args.sorting.lower() == 'scoredec':
            criterion = lambda p: -p[2]
        elif parsed_args.sorting.lower() == 'scoreinc':
            criterion = lambda p: p[2]
        else:
            print('Invalid sorting type.')
            return
        # bounds
        rmsdlowbound  = None
        rmsdupbound   = None
        scorelowbound = None
        scoreupbound  = 0
        if parsed_args.rmsdlim is not None:
            try:
                rmsdlowbound, rmsdupbound = process_limits(parsed_args.rmsdlim)
            except:
                print('Incorrectly specified RMSD limits.')
                return
        if parsed_args.scorelim is not None:
            try:
                scorelowbound, scoreupbound = \
                    process_limits(parsed_args.scorelim)
            except:
                print('Incorrectly specified score limits.')
                return
        self.text_buffer.seek(0, io.SEEK_END)
        writer = csv.writer(self.text_buffer, delimiter='\t')
        # I'm so sorry.
        writer.writerows(sorted([datapoint \
                                 for datapoint in data \
                                 if (((rmsdlowbound is None) or \
                                      rmsdlowbound < datapoint[1]) and \
                                     ((rmsdupbound is None) or \
                                      datapoint[1] < rmsdupbound) and \
                                     ((scorelowbound is None) or \
                                      scorelowbound < datapoint[2]) and \
                                     ((scoreupbound is None) or \
                                      datapoint[2] < scoreupbound))],
                                key=criterion))
    # Plots
    @continuous
    def do_plot_dir_rmsd_vs_score(self, arg):
        global PYROSETTA_ENV
        parser = argparse.ArgumentParser(
            description='For each PDB in a directory, plot the RMSDs vs a '
                        'particular file against their energy score.')
        parser.add_argument('in_dir',
                            action='store',
                            help='The directory to plot for.')
        parser.add_argument('in_file',
                            action='store',
                            help='The PDB to compare with for RMSD.')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs='*',
                            default=None,
                            help='Params files for the PDBs.')
        parser.add_argument('--rmsdlim',
                            nargs='*',
                            help='Limits on RMSD for the plot. If a side\'s '
                                 'limit is given as NONE, that side is '
                                 'unlimited.')
        parser.add_argument('--scorelim',
                            nargs='*',
                            help='Limits on score for the plot. If a side\'s '
                                 'limit is given as NONE, that side is '
                                 'unlimited.')
        parser.add_argument('--style',
                            dest='style',
                            action='store',
                            default='ro',
                            help='Matlab-type style to plot points with, like '
                                 '\'ro\' or \'b-\'.')
        self.do_plot_dir_rmsd_vs_score.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(arg.split())
        except SystemExit:
            return
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        params = process_params(parsed_args.params)
        data = set(zip(self.data_buffer.gather_rmsd(
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
                           params=params)))
        rmsdlowbound  = None
        rmsdupbound   = None
        scorelowbound = None
        scoreupbound  = 0
        if parsed_args.rmsdlim is not None:
            try:
                rmsdlowbound, rmsdupbound = process_limits(parsed_args.rmsdlim)
            except:
                print('Incorrectly specified RMSD limits.')
                return
        if parsed_args.scorelim is not None:
            try:
                scorelowbound, scoreupbound = \
                    process_limits(parsed_args.scorelim)
            except:
                print('Incorrectly specified score limits.')
                return
        # I'm so sorry.
        rmsds,scores = zip(*(datapoint \
                             for datapoint in data \
                             if (((rmsdlowbound is None) or \
                                  rmsdlowbound < datapoint[0]) and \
                                 ((rmsdupbound is None) or \
                                  datapoint[0] < rmsdupbound) and \
                                 ((scorelowbound is None) or \
                                  scorelowbound < datapoint[1]) and \
                                 ((scoreupbound is None) or \
                                  datapoint[1] < scoreupbound))))
        if self.settings['plotting']:
            plt.plot(rmsds, scores, parsed_args.style)
    @continuous
    def do_plot_dir_neighbors(self, arg):
        parser = argparse.ArgumentParser(
            description='Make a heatmap of how often two residues neighbor '
                        'each other in a given directory. You must specify '
                        'the range of residues over which to create the '
                        'heatmap.')
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
        self.do_plot_dir_neighbors.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(arg.split())
        except SystemExit:
            return
        params = process_params(parsed_args.params)
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        results = [np.array(result)
                   for result in \
                       self.data_buffer.gather_neighbors(
                           (os.path.join(parsed_args.in_dir, filename) \
                            for filename in filenames_in_in_dir \
                            if filename.endswith('.pdb')),
                           params=params)]
        if self.settings['plotting']:
            avg_matrix = np.mean(np.stack(results), axis=0)
            self.last_im = \
                plt.imshow(avg_matrix, cmap='jet', interpolation='nearest',
                           extent=[parsed_args.start_i, parsed_args.end_i,
                                   parsed_args.end_i, parsed_args.start_i],
                           aspect=1, vmin=0, vmax=1)
            plt.tick_params(axis='both', which='both',
                            top='off', bottom='off', left='off', right='off')
            self.do_prune_xticks('')
    @continuous
    def do_plot_neighbors_bar(self, arg):
        parser = argparse.ArgumentParser(
            description='Create a bar chart of a set of dirs for the average '
                        'long-distance neighbor rate among the PDBs in those '
                        'dirs. Set the labels using xticks.')
        parser.add_argument('dirs',
                            nargs='*',
                            help='Directories to chart.')
        parser.add_argument('--params',
                            nargs='*',
                            help='Path to params files; it is assumed that '
                                 'all PDBs need the same params. (If they '
                                 'don\'t, you can just list the params '
                                 'required by all of the PDBs together; '
                                 'Rosetta doesn\'t care.)')
        self.do_plot_neighbors_bar.__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            return
        params = process_params(parsed_args.params)
        values = []
        try:
            for pdbdir in parsed_args.dirs:
                filenames = os.listdir(pdbdir)
                results = [np.array(result) \
                           for result in \
                               self.data_buffer.gather_neighbors(
                                   (os.path.join(pdbdir, filename) \
                                    for filename in filenames \
                                    if filename.endswith('.pdb')),
                                   params=params)]
                avg_matrix = np.mean(np.stack(results), axis=0)
                # get a horizontally stacked version of matrix with middle
                # five diagonals removed
                N_REMOVED_DIAG = 5 # number of removed diagonals; must be odd
                stacked = np.hstack(
                    [np.hstack(avg_matrix[i][i+N_REMOVED_DIAG//2+1:] \
                               for i in range(avg_matrix.shape[0])),
                     np.hstack(avg_matrix[i][:max(i-N_REMOVED_DIAG//2,0)] \
                               for i in range(avg_matrix.shape[0]))])
                values.append(np.mean(stacked))
        except:
            print('Something weird went wrong; that\'s probably not a valid '
                  'set of dirs.')
            traceback.print_exc()
        if self.settings['plotting']:
            plt.barh(np.arange(len(values)), values, align='center')
    @continuous
    def do_plot_neighbors_comparison_bar(self, arg):
        parser = argparse.ArgumentParser(
            description='Plot a grouped bar chart where each group corresponds'
                        ' to a group of compared directories for different '
                        'PDBs, with each bar corresponding to a different PDB.'
                        ' The comparison itself is fraction nonnative '
                        'long-distance neighbors, with nativeness determined '
                        'by the presence of the contacts in a particular PDB '
                        'you specify (for each PDB in a directory group). Set '
                        'the labels using xticks.')
        parser.add_argument('dirs',
                            nargs='*',
                            help='Directories to chart. Give the directories '
                                 'in the same order their bars will appear in '
                                 'on the chart. So if you were comparing dirs '
                                 'for PDB1 and PDB2, you\'d list the dirs in '
                                 'the order "PDB1-dir1 PDB2-dir1 PDB1-dir2 '
                                 'PDB2-dir2."')
        parser.add_argument('--pdbs',
                            dest='pdbs',
                            action='store',
                            nargs='*',
                            help='Idealized PDBs to compare dirs against.')
        parser.add_argument('--params',
                            dest='params',
                            action='store',
                            nargs='*',
                            help='Path to params files; it is assumed that '
                                 'all PDBs need the same params. (If they '
                                 'don\'t, you can just list the params '
                                 'required by all of the PDBs together; '
                                 'Rosetta doesn\'t care.)')
        parser.add_argument('--scorelim',
                            nargs='*',
                            help='Limits on score for the plot. If a side\'s '
                                 'limit is given as NONE, that side is '
                                 'unlimited.')
        parser.add_argument('--colors',
                            dest='colors',
                            default=None,
                            help='List of Matlab-type colors to use for bars, '
                                 'as a smooshed-together list of letters. '
                                 'Example:  rgbcmyk')
        parser.add_argument('--missing-only',
                            dest='missingp',
                            action='store_true',
                            help='Catalog fraction of nonnative contacts only '
                                 'for contacts present in prototypes.')
        parser.add_argument('--nonnative',
                            dest='nonnativep',
                            action='store_true',
                            help='Catalog fraction nonnative contacts instead '
                                 'of native contacts.')
        parser.add_argument('--show-errors',
                            dest='errorsp',
                            action='store_true',
                            help='Add error bars (1 sigma).')
        self.do_plot_neighbors_comparison_bar \
            .__func__.__doc__ = parser.format_help()
        try:
            parsed_args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            return
        params = process_params(parsed_args.params)
        npdbs = len(parsed_args.pdbs)
        if len(parsed_args.dirs) % npdbs != 0:
            print('Incorrect dirs spec; number of dirs must be multiple of '
                  'number of pdbs. Try "help plot_neighbors_comparison_bar".')
            return
        pdbmatrices = [np.array(result) \
                       for result in \
                           self.data_buffer.gather_neighbors(parsed_args.pdbs,
                                                             params=params)]
        # parse bounds
        scorelowbound = None
        scoreupbound  = 0
        if parsed_args.scorelim is not None:
            try:
                scorelowbound, scoreupbound = \
                    process_limits(parsed_args.scorelim)
            except:
                print('Incorrectly specified score limits.')
                return
        # auxiliary function for the calculations
        def remove_diagonals(m, n_removed):
            '''Remove diagonals from the center of and flatten a matrix.'''
            return np.hstack(
                [np.hstack(m[i][i+n_removed//2+1:] \
                           for i in range(m.shape[0])),
                 np.hstack(m[i][:max(i-n_removed//2,0)] \
                           for i in range(m.shape[0]))])
        # do the calculations
        values = []
        if parsed_args.errorsp:
            errors = []
        else:
            errors = None
        try:
            for index, pdbdir in enumerate(parsed_args.dirs):
                prototype = pdbmatrices[index % npdbs]
                filenames = os.listdir(pdbdir)
                scores = self.data_buffer.gather_score(
                             (os.path.join(pdbdir, filename) \
                              for filename in filenames \
                              if filename.endswith('.pdb')),
                             params=params)
                results = [(np.array(result) - prototype)**2 \
                           for result in \
                               self.data_buffer.gather_neighbors(
                                   (os.path.join(pdbdir, filename) \
                                    for filename in filenames \
                                    if filename.endswith('.pdb')),
                                   params=params)]
                results = [m for i,m in enumerate(results) \
                           if (((scorelowbound is None) or \
                                scorelowbound < scores[i]) and \
                               ((scoreupbound is None) or \
                                scores[i] < scoreupbound))]
                # get horizontally stacked versions of matrices with middle
                # five diagonals removed
                N_REMOVED_DIAG = 5 # number of removed diagonals; must be odd
                linresults = [remove_diagonals(m, 5) for m in results]
                fracs = None
                if parsed_args.missingp:
                    linprototype = remove_diagonals(prototype, 5)
                    fracs = [np.sum(m*linprototype)/np.sum(linprototype) \
                             for m in linresults]
                else:
                    fracs = [np.sum(m)/m.size for m in linresults]
                if not parsed_args.nonnativep:
                    fracs = [1-m for m in fracs]
                values.append(np.mean(fracs))
                if parsed_args.errorsp:
                    errors.append(np.std(fracs))
        except:
            print('Something weird went wrong.')
            traceback.print_exc()
        if self.settings['plotting']:
            nbins = len(parsed_args.dirs) // len(parsed_args.pdbs)
            width = 0.5/nbins
            indices = np.array(
                          tuple(
                              itertools.chain.from_iterable(
                                  ((i+(j-npdbs/2)*width \
                                    for j in range(npdbs)) \
                                   for i in range(nbins)))))
            plt.bar(indices, values, yerr=errors,
                    color=parsed_args.colors, ecolor='k',
                    width=width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Interactive command-line interface for plotting and '
                      'analysis of batches of PDB files.')
    parser.add_argument('--background',
                        dest='backgroundp',
                        action='store_true',
                        help='Run given script in background and then '
                             'terminate. If no script is given, just do '
                             'nothing and terminate.')
    parser.add_argument('--continuous',
                        dest='continuousp',
                        action='store_true',
                        help='Re-run caching operations until they hit the '
                             'time limit or forever. By default, suppresses '
                             'plots.')
    parser.add_argument('script',
                        action='store',
                        nargs='?',
                        default=None,
                        help='.cmd file to run before entering interactive '
                             'mode.')
    parsed_args = parser.parse_args()
    PYROSETTA_ENV = PyRosettaEnv()
    OURCMDLINE = OurCmdLine()
    OURCMDLINE.settings['continuous_mode'] = parsed_args.continuousp
    OURCMDLINE.settings['plotting'] = not parsed_args.continuousp
    if MPIRANK != 0:
        DEVNULL = open(os.devnull, 'w')
        sys.stdout = DEVNULL
    if parsed_args.script is not None:
        with open(parsed_args.script) as f:
            OURCMDLINE.cmdqueue.extend(f.read().splitlines())
    if parsed_args.backgroundp:
        OURCMDLINE.cmdqueue.extend(['quit'])
    OURCMDLINE.cmdloop()
    if MPIRANK != 0:
        sys.stdout = STDOUT # defined at beginning of file
        DEVNULL.close()
