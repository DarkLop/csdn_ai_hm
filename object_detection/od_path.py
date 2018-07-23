
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
    previous_dir = os.path.split(dir_path)[0]
    if previous_dir not in sys.path:
        sys.path.append(previous_dir)
        print("Add research folder path: %s." % previous_dir)
    
    slim_dir = os.path.join(previous_dir, "slim")
    if os.path.exists(slim_dir) and os.path.isdir(slim_dir):
        if slim_dir not in sys.path:
            sys.path.append(slim_dir)
            print("Add slim folder path: %s." % slim_dir)
    
    _add_pathed = True	