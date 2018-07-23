
import os
import sys

_add_pathed = False

def add_path():
    global _add_pathed
    if _add_pathed is True:
        return

    file_name = __file__
    real_path = os.path.realpath(file_name)
    dir_path = os.path.dirname(real_path)
    if dir_path not in sys.path:
        sys.path.append(dir_path)
        print("add path: %s." % dir_path)
    _add_pathed = True
