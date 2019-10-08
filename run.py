import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as sio
import glob2
import getopt

def translational(data, origin_bp):
    # data format: num_bp x num_coord x t
    # origin_bp - specifies which body point to make the origin
    return ( np.copy(data - data[origin_bp,:,:]), np.copy(data[origin_bp,:,:]) )

def rotational(data, axis_bp, align_to_posY=True):
    # rotate axis to be vertical; only works with 2 dimensions as of right now
    # data format: num_bp x (X_coord, Y_coord) x t
    # angle_list: angle of rotation from the vertical per frame
    rot_data = np.copy(data)
    num_bp = rot_data.shape[0]
    axis_vector = rot_data[axis_bp,:,:]
    if align_to_posY:
        angle_list = np.sign(axis_vector[0,:]) * np.pi/2 - np.arctan( axis_vector[1,:]/axis_vector[0,:] ) # angle rotated per frame
    else:
        angle_list = -1*np.sign(axis_vector[0,:]) * np.pi/2 - np.arctan( axis_vector[1,:]/axis_vector[0,:] ) # angle rotated per frame
    # rotate each body point
    for i in range(num_bp):
        rot_data[i,:,:] = Rotate(rot_data[i,:,:], angle_list)
    return (rot_data, angle_list)

# helper tool function
def Rotate(data, angle):
    return np.einsum('ijk,jk ->ik', np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]]), data)

def ajust_data(filepath, num_bp=30, origin_bp=2, axis_bp=1, align_to_posY=True):
    # specified parameter
    # num_bp - number of body point labeled
    # origin_bp -  origin body point to center animal
    # axis_bp -  other body point to form an axis w/ origin body point
    # delete_bp - should always delete origin bp; can add other bp to list

    data_path_list = glob2.glob(filepath+'/*')
    delete_bp=(origin_bp)
    total_frame = 0
    for data_path in data_path_list:
        dir_name = os.path.basename(data_path) # specific directory name in data folder
        
        # convert body point h5 to numpy
        bp_path = glob2.glob(data_path+'/*.h5')[0]
        # bp_h5data = pd.read_hdf(bp_path)
        store = pd.HDFStore(bp_path)
        df = store['/df_with_missing']
        # bp_data = bp_h5data[ bp_h5data.keys().levels[0][0] ].values # converts h5 to npy
        bp_data = df.to_numpy()
        store.close()
        # reformat numpy body point data
        num_frame = bp_data.shape[0]
        bp_data = np.delete( bp_data.reshape( num_frame,num_bp,-1 ), obj=-1, axis=2 ) # reformats data and takes out last prob varaiable
        bp_data = np.swapaxes(bp_data.T,0,1) # num_bp x num_coord x t
        # translate data w/ respect to origin
        (bp_data, trans_data) = translational(bp_data, origin_bp)
        # rotate data w/ respect to body axis
        (bp_data, rot_data) = rotational(bp_data, axis_bp, align_to_posY)
        # visualize transformed body points
        # visualize.ant_bp_graph(bp_data, frame=600)
        
        # delete unwanted body points
        bp_mod = np.delete( bp_data,delete_bp, 0)
        # reshape body point to [N_frame x features] for spectrogram
        num_relevant_bp, num_axis = bp_mod.shape[0], bp_mod.shape[1]
        bp_data_mod = bp_mod.reshape( num_relevant_bp*num_axis,-1 ).T 
        # subtract mean for each bp for spectrogram analysis
        bp_data_mod = bp_data_mod - np.mean(bp_data_mod, axis=0)
        # accumulate number of frames
        total_frame += num_frame
        # save data
        np.save(data_path+"/BP_"+dir_name, bp_data)
        np.save(data_path+"/TRANS_"+dir_name, trans_data)
        np.save(data_path+"/ROT_"+dir_name, rot_data)
        sio.savemat( data_path+"/MAT_"+dir_name+".mat",{"projections":bp_data_mod} )

        print("number of frames: ", num_frame)
        print("datapath: ", data_path)
        print("directory: ", dir_name)
        print("body point data: ", bp_data.shape)
        print("spectrogram data: ", bp_data_mod.shape)
        print("*********************************************")
        
    print("total number of frames: ", total_frame)

if __name__ == '__main__':
    # default values
    num_bp, origin_bp, axis_bp = 30, 2, 1
    # response for specific flags
    if "-h" in sys.argv:
        print("-f | file path to the deeplabcut folder(REQUIRED)")
        print("-b | number of bodypoint")
        print("-o | origin of bodypoint index")
        print("-a | axis body point w/ respect to origin")
        print("-y | align to positive y axis (True/False)")
        sys.exit(2)
    if "-f" not in sys.argv:
        print(":: no filepath to data")
        sys.exit(2)
    # get flags and argument
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'f:b:o:a:y')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for flag, arg in optlist:
        if flag == "-f": 
            if os.path.exists(arg):
                filepath = arg
            else:
                print(":: filepath does not exist")
                sys.exit(2)
        elif flag == "-b": num_bp = int(arg)
        elif flag == "-o": origin_bp = int(arg)
        elif flag == "-a": axis_bp = int(arg)
        elif flag == "-y": align_to_posY = bool(arg)
    print("Number of Bodypoint: ", num_bp)
    print("Origin Bodypoint: ", origin_bp)
    print("Axis Bodypoint: ", axis_bp)
    print("Align to Positive Y: ", align_to_posY)
    # modify data
    ajust_data(filepath, num_bp, origin_bp, axis_bp, align_to_posY)

    
