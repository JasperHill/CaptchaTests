from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import shutil
import random
import re
import os
import io

training_labels = []
testing_labels = []

training_images = []
testing_images = []

cwd = os.getcwd()
data_path = os.path.join(cwd,'captcha_samples')
files = os.listdir(data_path)

training_path = os.path.join(data_path,'training_samples')
testing_path = os.path.join(data_path,'testing_samples')

##split total dataset in half
for file_name in files:
    r = random.gauss(0,1)

    if re.search('png',file_name) is not None:
        if r <= 0:
            shutil.move(os.path.join(data_path,file_name),training_path)

        else:
            shutil.move(os.path.join(data_path,file_name),testing_path)

    elif re.search('jpg',file_name) is not None:
        if r<= 0:
            shutil.move(os.path.join(data_path,file_name),training_path)

        else:
            shutil.move(os.path.join(data_path,file_name),testing_path)




