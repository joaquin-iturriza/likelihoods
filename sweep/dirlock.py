"""
Exclusive lock shared by the jobs of one sweep, built on an atomic mkdir.

fcntl.flock is not available everywhere a sweep directory lives: EOS has no
advisory locks, and on CC-IN2P3's /sps (NFS) some compute nodes answer every
flock with ENOLCK ("No locks available"), which killed 4 of 20 trials at their
first suggest(). mkdir of an existing directory fails atomically on every one
of these filesystems, so the lock is the directory <path>.lockdir; the holder
writes host and pid into it. A lock older than STALE_S (a job killed while
holding it; a holder needs seconds) is broken by renaming it away, which only
one waiter can do.
"""

import os
import socket
import time
from contextlib import contextmanager

STALE_S = 1800
POLL_S = 1.0


@contextmanager
def dir_lock(path: str):
    lock = path + '.lockdir'
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock).st_mtime
            except FileNotFoundError:
                continue
            if age > STALE_S:
                stale = f"{lock}.stale.{socket.gethostname()}.{os.getpid()}"
                try:
                    os.rename(lock, stale)
                    _rmtree(stale)
                except OSError:
                    pass
                continue
            time.sleep(POLL_S)
    try:
        with open(os.path.join(lock, 'owner'), 'w') as f:
            f.write(f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n")
        yield
    finally:
        _rmtree(lock)


def _rmtree(d):
    for name in os.listdir(d):
        os.remove(os.path.join(d, name))
    os.rmdir(d)
