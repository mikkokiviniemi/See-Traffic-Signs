import cv2
import shutil
import argparse
import os
import random

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
    os.makedirs(target_path + "train")
    os.makedirs(target_path + "test")
    os.makedirs(target_path + "evaluation")
except:
    print("Target path already exists")
    exit(1)

print("Starting..")

files = [f for f in os.listdir(source_path) if f[-4:] == ".jpg" ]

for img_file in files:

    img_path = source_path + img_file

    num = random.random()

    if num < 0.1:
        target_img_path = target_path + "test/" + img_file
    elif num < 0.2:
        target_img_path = target_path + "evaluation/" + img_file
    else:
        target_img_path = target_path + "train/" + img_file

    shutil.copyfile(img_path, target_img_path)
    shutil.copyfile(img_path[0:-4] + ".xml", target_img_path[0:-4] + ".xml")
