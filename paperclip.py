#!/opt/sw/packages/gcc-4.8/python/3.5.2/bin/python3

#### paperclip.py
#### Plotting and Analyzing Proteins Employing Rosetta CLI Program
#### by Michael Szegedy, 2017 Oct
#### for Khare Lab at Rutgers University

'''This is an interactive suite for generating a variety of plots based on data
obtained from Rosetta and MPRE (mszpyrosettaextension). Mostly the plots
revolve around data that I've had to collect for research, or personal
interest. Most plotting commands are based on Matlab commands.
'''

import os, sys
import fcntl, hashlib, json, subprocess, time
import cmd
import pyrosetta as pr
import mszpyrosettaextension as mpre
from pyrosetta.rosetta.core.scoring import all_atom_rmsd
import matplotlib
# matplotlib.use("Agg") # otherwise lack of klab display breaks program
import matplotlib.pyplot as plt

PYROSETTA_ENV = None

### Decorators

def needs_pr_init(f):
    """Makes sure PyRosetta gets initialized before f is called."""
    def decorated(*args, **kwargs):
        global PYROSETTA_ENV
        if not PYROSETTA_ENV.initp:
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
            PYROSETTA_ENV.set_scorefxn()
        return f(*args, **kwargs)
    return decorated

### Housekeeping classes

class PyRosettaEnv():
    """Stores stuff relating to PyRosetta, like whether pr.init() has been
    called."""
    def __init__(self):
        self.initp = False
        self.scorefxn = None
    @needs_pr_init
    def set_scorefxn(name=None, patch=None):
        """Sets the scorefxn and optionally applies a patch. Defaults to
        get_fa_scorefxn() with no arguments."""
        if name is None:
            self.scorefxn = pr.get_fa_scorefxn()
        else:
            self.scorefxn = pr.create_score_function(name)
        if patch:
            self.scorefxn.apply_patch_from_file(patch)

### Main class

class OurCmdLine(cmd.Cmd):
    """Singleton class for our interactive CLI."""
