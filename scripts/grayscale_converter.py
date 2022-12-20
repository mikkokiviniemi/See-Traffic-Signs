import cv2
import shutil
import argparse
import os


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-p', '--path', help='Path of the images to be converted')
arg_parser.add_argument('-t', '--target', help='Path of the converted images')
ns = arg_parser.parse_args()

def clean_path(path):
    new_path = path.replace("\\","/")
    if new_path[-1:] != "/":
        new_path = new_path + "/"
    return new_path

if ns.path and ns.target:
    source_path = clean_path(ns.path)
    target_path = clean_path(ns.target)
else:
    exit(1)

try:
    os.makedirs(target_path)
except:
    print("Target path already exists")
    exit(1)

print("Starting conversion..")

success = 0
failed = 0
files = [f for f in os.listdir(source_path) if str.lower(f[-4:]) == ".jpg" ]

for img_file in files:
    try:
        img_path = source_path + img_file
        target_img_path = target_path + img_file
        
        img = cv2.imread(img_path)
        grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(target_img_path, grayscale_img)
        shutil.copyfile(img_path[0:-4] + ".xml", target_img_path[0:-4] + ".xml")
        success += 1
    except:
        failed += 1

print("Succesfully converted {} images to grayscale".format(success))
print("Failed to convert {} images".format(failed))

     