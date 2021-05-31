import os
import shutil

subfolders = os.listdir("miniImageNet/images")

print(subfolders)

for folder in subfolders:
    files = os.listdir(os.path.join("miniImageNet/images", folder))
    for file in files:
        shutil.move(os.path.join("miniImageNet/images/"+folder, file), "miniImageNet/images/"+file)
    os.rmdir(os.path.join("miniImageNet/images", folder))