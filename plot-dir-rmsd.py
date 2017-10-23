#!/opt/sw/packages/gcc-4.8/python/3.5.2/bin/python3

#### by Michael Szegedy, 2017 Jul
#### for Khare Lab at Rutgers University

#### This script makes an all-atom RMSD plot, with a datapoint for each PDB in
#### a directory compared to a single given PDB.

import os, sys
import argparse, errno, fcntl, hashlib, json, subprocess, time
from pyrosetta import *
import mszpyrosettaextension as mpre
from pyrosetta.rosetta.core.scoring import all_atom_rmsd
import matplotlib
# matplotlib.use("Agg") # otherwise lack of klab display breaks program
import matplotlib.pyplot as plt

pyrosetta_initialized_p = False
scorefxn = None
base_pose = None
parsed_args = None

def get_data_from_file(file_path, mtime, params=None):
    # I know it's ugly, but I really prefer it to passing every damn thing into
    # this function; honestly I should actually just rewrite this whole thing
    # with OO stuff, but that's really unnecessary for such a simple program.
    #
    # NOTE FROM THE FUTURE: It's no longer a simple program. Still not rewriting
    #  it.
    global pyrosetta_initialized_p
    global scorefxn
    global base_pose
    global parsed_args
    if not pyrosetta_initialized_p:
        init()
        scorefxn = get_fa_scorefxn()
        if parsed_args.params:
            base_pose = mpre.pose_from_file_with_params(parsed_args.in_file,
                                                        parsed_args.params)
        else:
            base_pose = pose_from_file(parsed_args.in_file)
        pyrosetta_initialized_p = True
    pose = None
    if parsed_args.params:
        pose = mpre.pose_from_file_with_params(file_path,
                                               parsed_args.params)
    else:
        pose = pose_from_file(file_path)
    rmsd = all_atom_rmsd(base_pose, pose)
    score = scorefxn(pose)
    # mtime is provided as an argument
    return [rmsd, score, mtime]

def update_cache_from_file(filename):
    # I should probably move get_data_from_file into here, but maybe I wanna
    # reuse it somewhere else? Modularity is generally a good thing.
    global parsed_args
    global data
    cachemtime = None
    try:
        cachemtime = data[filename][2]
    except KeyError:
        pass
    file_path = os.path.join(parsed_args.in_dir, filename)
    newmtime = os.path.getmtime(file_path)
    if cachemtime is None or newmtime > cachemtime:
        # get updated data for data[filename]
        data_update = get_data_from_file(file_path,
                                         newmtime,
                                         parsed_args.params)
        with open(cache_filename, 'w+') as cache_file:
            # try to get lock on cache file
            while True:
                try:
                    fcntl.flock(cache_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            # now that we have a lock, load the newest version of the cache
            try:
                data = json.loads(cache_file.read())
            except ValueError:
                pass
            # gotta get cachemtime again; don't wanna overwrite newer
            # analysis
            cachemtime = None
            try:
                cachemtime = data[filename][2]
            except KeyError:
                pass
            if cachemtime is None or newmtime > cachemtime:
                data[filename] = data_update
                cache_file.write(json.dumps(data))
            fcntl.flock(cache_file, fcntl.LOCK_UN)

## parse arguments
parser = argparse.ArgumentParser(
    description="Create an all-atom RMSD plot. By default, excludes PDBs "
                "with positive scores.")
parser.add_argument("--in-file",
                    dest="in_file",
                    action="store",
                    help="PDB file to compare other files to.")
parser.add_argument("--in-dir",
                    dest="in_dir",
                    action="store",
                    default=os.getcwd(),
                    help="Directory full of PDB files to compute RMSD of.")
parser.add_argument("--out-file",
                    dest="out_file",
                    action="store",
                    help="Filename to save the output at (type may be SVG, "
                         "PS, EPS, PDF, or PNG).")
parser.add_argument("--max-score",
                    dest="max_score",
                    action="store",
                    type=float,
                    default=0,
                    help="Score above which to filter out data points.")
parser.add_argument("--suppress-plot",
                    dest="save_plot_p",
                    action="store_false",
                    help="Update cache only; don't save plot.")
parser.add_argument("--suppress-update",
                    dest="update_cache_p",
                    action="store_false",
                    help="Save plot only; don't update cache.")
parser.add_argument("--copy-cache",
                    dest="cache_to_copy",
                    action="store",
                    default=None,
                    help="Initialize the cache from the given file before "
                         "updating it.")
parser.add_argument("--continuous-mode",
                    dest="continuous_mode_p",
                    action="store_true",
                    help="Activate continuous mode, where the program checks "
                         "the directory for new files after every cache "
                         "update. When combined with --suppress-plot, the "
                         "program will keep checking periodically until it is"
                         " terminated from the outside.")
parser.add_argument("--params",
                    dest="params",
                    action="store",
                    nargs=argparse.REMAINDER,
                    default=[],
                    help="Params files, if any. It is assumed that all input "
                         "files require the same params.")
parsed_args = parser.parse_args()

## get data from cache, if applicable
params_list = list(parsed_args.params)
params_list.sort()
hash_args = (parsed_args.in_file, parsed_args.in_dir) + tuple(params_list)
hash_fun = hashlib.md5()
hash_fun.update(str(' '.join(hash_args)).encode())
cache_filename = ".plot_dir_rmsd_cache_"+hash_fun.hexdigest()
data = {} # maps filenames to a list of rmsd, score, and file mtime at reading
if parsed_args.cache_to_copy is not None:
    try:
        with open(parsed_args.cache_to_copy, 'r') as cache_file:
            data = json.loads(cache_file.read())
    except IOError:
        print("Warning: Supplied cache file for copying does not exist! "
              "Starting with empty cache.")
    except ValueError:
        print("Warning: Supplied cache file for copying is empty!")
else:
    try:
        with open(cache_filename, 'r') as cache_file:
            data = json.loads(cache_file.read())
    except (IOError, ValueError):
        pass

## get data for files not in cache (or outdated in cache)
if parsed_args.update_cache_p:
    if not parsed_args.continuous_mode_p:
        filenames_in_in_dir = os.listdir(parsed_args.in_dir)
        # remove files from cache that no longer exist
        for filename in data.keys():
            if filename not in filenames_in_in_dir:
                del data[filename]
        # update cache
        for filename in [x for x in filenames_in_in_dir if x.endswith(".pdb")]:
            update_cache_from_file(filename)
    else:
        old_in_dir_image = subprocess.check_output(['ls', '-l',
                                                    parsed_args.in_dir])
        no_change = False
        while not no_change or not parsed_args.save_plot_p:
            filenames_in_in_dir = os.listdir(parsed_args.in_dir)
            # remove files from cache that no longer exist
            for filename in data.keys():
                if filename not in filenames_in_in_dir:
                    del data[filename]
            # update cache
            for filename in [x \
                             for x in filenames_in_in_dir \
                             if x.endswith(".pdb")]:
                update_cache_from_file(filename)
            new_in_dir_image = subprocess.check_output(['ls', '-l',
                                                        parsed_args.in_dir])
            no_change = old_in_dir_image == new_in_dir_image
            old_in_dir_image = new_in_dir_image
            time.sleep(1)

## make plot
if parsed_args.save_plot_p:
    # there's gotta be a more idiomatic way to do this
    filtered_values = [value \
                       for value in data.values() \
                       if value[1] <= parsed_args.max_score]
    if len(filtered_values) > 1:
        rmsds  = [value[0] for value in filtered_values]
        scores = [value[1] for value in filtered_values]
        plt.plot(rmsds, scores, "ro")
        plt.xlabel("RMSD")
        try:
            plt.ylabel(scorefxn.get_name() + " score")
        except AttributeError:
            plt.ylabel("talaris2014 score")
        plt.ylim(ymin=min(scores), ymax=parsed_args.max_score)
        plt.savefig(parsed_args.out_file,
                    format=parsed_args.out_file.split('.')[-1].lower())
    else:
        print("Nothing to plot! Only " + str(len(filtered_values)) + \
              " valib PDB(s) found.")
