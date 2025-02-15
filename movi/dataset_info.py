# cmdline: python3 dataset_info.py F_v3d_Subject_*.mat
# argv[0] = dataset_info.py
# argv[1] = Subject data file

from utils import *

import pandas as pd
import numpy as np
import sys

v3d_filename = sys.argv[1]

sample = mat2dict(v3d_filename) # from utils.py
sample.keys()
move = sample['move']
move =dict2ntuple(move) # from utils.py

subject_df = pd.DataFrame.from_dict(sample)
subject = subject_df['subject']['id']

flags120 = move.flags120
motions = move.motions_list
motion_dict = {}
total_frames = 0
frames_list = []

rm = [rm for rm in motions if "_rm" in rm]

for i in range(0, len(motions)):
    df_name = motions[i]
    start, end = flags120[i]
    motion_dict[df_name] = move.virtualMarkerLocation[start:end+1]
    frames_list.append(motion_dict[df_name].shape[0])
    total_frames = total_frames + motion_dict[df_name].shape[0]
    

with open("dataset_info.txt", 'a') as csv_file:
        csv_file.write(subject)
        csv_file.write(' ')
        csv_file.write(str(total_frames))
        csv_file.write(' ')
        csv_file.write(str(min(frames_list)))
        csv_file.write(' ')
        csv_file.write(str(max(frames_list)))
        csv_file.write(' ')
        csv_file.write(rm[0])
        csv_file.write('\n')
