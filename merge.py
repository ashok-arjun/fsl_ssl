import os
import shutil

"""miniImageNet"""

root = "./filelists/miniImagenet/images"

"""tieredImageNet"""

root = "./filelists/tieredImagenet/images/train" # repeat with val, test


subdirectories = [root + x for x in os.listdir(root)]

for subdir in subdirectories:
    files = [subdir + "/" + x for x in os.listdir(subdir)]
    for file in files:
        shutil.move(file, root)
    os.rmdir(subdir)
