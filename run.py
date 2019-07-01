import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
# supplement
# from tqdm import tqdm, tqdm_notebook
import scipy.io as sio
import glob2
# tools
from tools import transform, visualize

def ajust_data(num_bp=30, origin_bp=2, axis_bp=1):
	# specified parameter
	# num_bp - number of body point labeled
	# origin_bp -  origin body point to center animal
	# axis_bp -  other body point to form an axis w/ origin body point
	# delete_bp - should always delete origin bp; can add other bp to list

	data_path_list = glob2.glob('data/*')
	delete_bp=(origin_bp)
	for data_path in data_path_list:
	    dir_name = os.path.basename(data_path) # specific directory name in data folder
	    
	    # convert body point h5 to numpy
	    bp_path = glob2.glob(data_path+'/*.h5')[0]
	    bp_h5data = pd.read_hdf(bp_path)
	    bp_data = bp_h5data[ bp_h5data.keys().levels[0][0] ].values # converts h5 to npy
	    # reformat numpy body point data
	    num_frame = bp_data.shape[0]
	    bp_data = np.delete( bp_data.reshape( num_frame,num_bp,-1 ), obj=-1, axis=2 ) # reformats data and takes out last prob varaiable
	    bp_data = np.swapaxes(bp_data.T,0,1) # num_bp x num_coord x t
	    # get camera position
	    cam_path = glob2.glob(data_path+'/*.npy')[0]
	    cam_data = np.load(cam_path)
	    # translate data w/ respect to origin
	    (bp_data, trans_data) = transform.translational(bp_data, origin_bp)
	    # rotate data w/ respect to body axis
	    (bp_data, rot_data) = transform.rotational(bp_data, axis_bp)
	    # visualize transformed body points
	    # visualize.ant_bp_graph(bp_data, frame=600)
	    
	    # delete unwanted body points
	    bp_mod = np.delete( bp_data,delete_bp,0 )
	    # reshape body point to [N_frame x features] for spectrogram
	    num_relevant_bp, num_axis = bp_mod.shape[0], bp_mod.shape[1]
	    bp_data_mod = bp_mod.reshape( num_relevant_bp*num_axis,-1 ).T 
	    # subtract mean for each bp for spectrogram analysis
	    bp_data_mod = bp_data_mod - np.mean(bp_data_mod, axis=0)
	    
	    # save data
	    np.save(data_path+"/BP_"+dir_name, bp_data)
	    np.save(data_path+"/TRANS_"+dir_name, trans_data)
	    np.save(data_path+"/ROT_"+dir_name, rot_data)
	    sio.savemat( data_path+"/MAT_"+dir_name+".mat",{"projections":bp_data_mod} )

	    print("number of frames: ", num_frame)
	    print("datapath: ", data_path)
	    print("directory: ", dir_name)
	    print("body point data: ", bp_data.shape)
	    print("camera data: ", cam_data.shape)
	    print("spectrogram data: ", bp_data_mod.shape)
	    print("*********************************************")

if __name__ == '__main__':
	if len(sys.argv) > 1:
		num_bp, origin_bp, axis_bp=sys.argv[1], sys.argv[2], sys.argv[3]
	else:
		num_bp, origin_bp, axis_bp=30, 2, 1
	ajust_data(num_bp, origin_bp, axis_bp)

    
