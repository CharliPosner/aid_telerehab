# cmdline: python3 movi2img.py F_v3d_Subject_SUBJECT.mat SUBJECT
# argv[0] = movi2img.py
# argv[1] = Subject data file
# argv[2] = Subject number

from utils import *
from PIL import Image
import pandas as pd
import numpy as np
import os
import glob

##### COMMAND LINE INPUTS #####

main_dir = os.getcwd()
movi_dir = main_dir + '\\F_Subjects_data\\'
sub_dirs = glob.glob(main_dir + '\\*\\', recursive = True)
frames = 120 # 120 frames taken from middle of mocap sequence

#################### NORMALISE SKELETON #############################

blazepose_idx = [0, 14, 6, 10, 2, 15, 7, 12, 4, 11, 3, 13, 5, 9, 1, 16, 8]

def normalise_skeleton(motion_dict):
    norm_dict = {}
    for motion in motion_dict:
        joint_positions = motion_dict[motion]
        norm_motion = np.empty([len(joint_positions), 17, 3])
        sorted_motion = np.empty([len(joint_positions), 17, 3])
        
        # get coords of middle hip point at frame i 
        origin = joint_positions[:, 1, :] ## origin: (frame, 3)
        joint_positions = np.delete(joint_positions, 1, axis=1)
        
        # for each frame in the sequence
        for frame in range(len(joint_positions)):
            
            # normalise wrt. new origin
            norm_motion[frame, :, :] = np.subtract(joint_positions[frame, :, :], origin[frame, :])
            
            # get mid-shoulder point for height normalisatioon
            left_shoulder = norm_motion[frame, 6, :]
            right_shoulder = norm_motion[frame, 14, :]
            shoulder_vector = np.subtract(left_shoulder, right_shoulder)
            shoulder_midpoint = left_shoulder - np.divide(shoulder_vector, 2)
            
            # normalise skeleton such that DISTANCE (magnitude) from mid-shoulders to mid-hips is 1
            ref_length = np.linalg.norm(np.absolute(shoulder_midpoint))
            norm_motion[frame, :, :] = np.divide(norm_motion[frame, :, :], ref_length)
            
            # fix axes
            sorted_motion[frame, :, 0] = [norm_motion[frame, i, 1] for i in blazepose_idx]
            sorted_motion[frame, :, 1] = [norm_motion[frame, i, 2] for i in blazepose_idx]
            sorted_motion[frame, :, 2] = [norm_motion[frame, i, 0] for i in blazepose_idx]
            
        norm_dict[motion] = sorted_motion
        
    return norm_dict


##### MATLAB FILE TO PYTHON DICTIONARY #####

for rootdir, dirs, files in os.walk(movi_dir):
	for mat_file in files:
		print(mat_file)
		subject = mat_file.split('_')[3].split('.')[0]

		sample = mat2dict(os.path.join(movi_dir, mat_file)) # from utils.py
		sample.keys()
  
		move = sample['move']
		move =dict2ntuple(move) # from utils.py
		subject_df = pd.DataFrame.from_dict(sample)

		flags120 = move.flags120
		motions = move.motions_list
		motion_dict = {}
		norm_dict = {}

		for i in range(0, len(motions)):
			df_name = motions[i]
			start, end = flags120[i]
			motion_dict[df_name] = np.delete(move.virtualMarkerLocation[start:end+1], [2, 3], axis=1)

		##### DEAL WITH OUTLIERS #####

		for motion in motions:
			outlier_mask = np.all(motion_dict[motion] == 0, axis=2)
			broken_joint_list = np.where(np.any(outlier_mask, axis=0))[0]

			for broken_joint in broken_joint_list:
				for xyz in range(3):
					frame_number_vec = np.arange(outlier_mask.shape[0])
					good_frames = np.logical_not(outlier_mask[:, broken_joint])
					bad_frames = outlier_mask[:, broken_joint]

					fp = motion_dict[motion][good_frames, broken_joint, xyz]
					xp = frame_number_vec[good_frames]
					x = frame_number_vec[bad_frames]

					if (len(fp) == 0):
						fp = [0] * frames
						xp = [0] * frames
      
					fixed_frames = np.interp(x, xp, fp)
					motion_dict[motion][bad_frames, broken_joint, xyz] = fixed_frames

		norm_dict = normalise_skeleton(motion_dict)
		print(norm_dict['kicking'].shape)


		##### ACTIONS TO IMAGES ######

		for motion in motions:

			# a) take the middle N frames
			full_motion = norm_dict[motion]
			start_idx = (len(full_motion) // 2) - (frames // 2)
			end_idx = (len(full_motion) // 2) + (frames // 2)
			motion_slice = full_motion[start_idx: end_idx]

			# b) apply RGB scaling
			R = motion_slice[..., 0]
			G = motion_slice[..., 1]
			B = motion_slice[..., 2]

			R = ((R-R.min())/(R.max()-R.min()) * 255).astype('uint8')
			G = ((G-G.min())/(G.max()-G.min()) * 255).astype('uint8')
			B = ((B-B.min())/(B.max()-B.min()) * 255).astype('uint8')

			RGB = np.dstack((R,G,B))

			# c) reshape into ResNet input (224, 224, 3)
			image = Image.fromarray(RGB)
			image = image.resize((224, 224))

			# d) save as "SUBJECT_ACTION.png"
			action = motion.translate({ord(i): None for i in '/_-'}) # sanitise inputs

			if (action == 'jumpingjacks'):
				action = 'jumpingjack'

			file_name = f'{subject}_{action}.png'
			action_folder = f'D:\\Misc\\MoVi\\movi2img\\{action}\\'
			
			if action_folder not in glob.glob('D:\\Misc\\MoVi\\movi2img\\*\\'):
				os.makedirs(action_folder)

			file_path = os.path.join(main_dir, action_folder)
			image.save(os.path.join(file_path, file_name))
			


####################################################################

            
            
