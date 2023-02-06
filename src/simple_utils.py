import pickle
import torch
import filelock
import logging

from os import listdir, mkdir
from os.path import isfile, join, dirname, basename, isdir

logging.getLogger("filelock").setLevel(logging.ERROR)

def list_files(path: str):
    return [f for f in listdir(path) if isfile(join(path, f))]

def load_pickle(fname, lock_dir='locks'):
    lockfname = fname + '.lock'
    if lock_dir is not None:
        lock_path = join(dirname(lockfname), lock_dir)
        if not isdir(lock_path):
            mkdir(lock_path)
        lockfname = join(lock_path, basename(lockfname))
    lock = filelock.FileLock(lockfname)

    try:
        with lock.acquire(timeout=10):
            try:
                with open(fname, 'rb') as f:
                    result = pickle.load(f)
            except FileNotFoundError:
                print("file {} does not exist".format(fname))
    except filelock.Timeout:
        print("failed to read in time")
    except Exception as e:
        print(e)

    return result

def dump_pickle(obj, fname, lock_dir='locks'):
    lockfname = fname + '.lock'
    if lock_dir is not None:
        lock_path = join(dirname(lockfname), lock_dir)
        if not isdir(lock_path):
            mkdir(lock_path)
        lockfname = join(lock_path, basename(lockfname))

    try:
        with filelock.FileLock(lockfname, timeout=10):
            with open(fname, 'wb') as f:
                pickle.dump(obj, f)
    except filelock.Timeout:
        print("failed to write in time")
    except Exception as e:
        print(e)
