import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import torch
import time
import os


class MorletTrans:
    def __init__(self, w_0, N_f, f_min, f_max, fr=50):
        self.w_0 = w_0
        self.N_f = N_f
        self.f_min = f_min
        self.f_max = f_max
        self.f_array = self.freq(np.arange(1, N_f+1))
        self.fr = fr
        self.data_length = 1000 # Random length initially
    def morlet(self, eta):
        return np.pi**(-0.25)*np.exp(1j*self.w_0*eta)*np.exp(-0.5*eta**2)
    def s(self, f):
        return (self.w_0+np.sqrt(2+self.w_0**2))/(4*np.pi*f)
    def freq(self, i):
        return self.f_max*2**(-1*(i-1)/(self.N_f-1)*np.log2(self.f_max/self.f_min))
    def C(self, s_cons):
        return ((np.pi**(-0.25))/np.sqrt(2*s_cons))*np.exp(((self.w_0-np.sqrt(self.w_0**2+2))**2)/4)
        # return ((np.pi**(-0.25))/np.sqrt(2*s_cons))*np.exp(1/(((self.w_0-np.sqrt(self.w_0**2+2))**2)*4))
    def run(self, data):
        num_bp = data.shape[0]
        num_t = data.shape[-1]
        self.data_length = num_t
        num_dim = data.shape[1]
        t = np.arange(num_t)
        power = np.zeros((num_bp, num_dim, self.f_array.size, num_t)) 
        for bp_i, bp in enumerate(range(num_bp)):
            for axis in range(num_dim):
                if num_t % 2 == 0:
                    y = data[bp_i,axis,:-1] # if even
                else:
                    y = data[bp_i,axis,:] # if odd
                for f_idx, f in enumerate(self.f_array):
                    s_cons = self.s(f)
                    # CWT Transform
                    transform = (1/np.sqrt(s_cons))*np.convolve(y, self.morlet(t/s_cons), mode="same") 
                    power[bp_i,axis,f_idx,:] = (1/self.C(s_cons))*np.absolute(transform)
        return power
    def plot(self, power_bp, save_as_pdf, t_start=0, t_end=100, cmap="OrRd"):
        fig, axes = plt.subplots(power_bp.shape[0], power_bp.shape[1], figsize=(20,115)) # (height, width) 
        # CREATE SUBPLOT
        for bp_i in tqdm(range(power_bp.shape[0])):
            for axis in range(power_bp.shape[1]):
                mesh = axes[bp_i,axis].pcolormesh(power_bp[bp_i,axis,:,t_start:t_end])# , edgecolors='face', linewidths=5
                axes[bp_i,axis].set_title("Wavelet Spectrogram Body Part " + str(bp_i) + " with Axis " + str(axis))
                axes[bp_i,axis].set_xlabel("Time (s)")
                axes[bp_i,axis].set_ylabel("Frequency (Hz)")
                # axes[bp_i,axis].set_yticklabels(np.around(self.f_array, decimals=2), minor=False)
                # fig.colorbar(mesh)
        plt.tight_layout()
        plt.show()
        # SAVE FIGURE AS PDF
        if(save_as_pdf == True):
            f = open("fig/" + 'RENAME_ME' + ".pdf","x")
            fig.savefig("fig/" + 'RENAME_ME' + ".pdf")
            f.close() 
        plt.close(fig)
        
class Morlet():
    def __init__(self, ):
        self.filename = ""
        self.df_data = None 
        self.data = None
        self.power_bp = None
        self.trans_data = None
        self.rot_data = None
    def import_data(self, data_path):
        # Import Data
        self.filename=os.path.splitext(os.path.basename(data_path))[0]
        df = pd.read_hdf(data_path)
        self.df_data = df[df.keys().levels[0][0]].values
        pass
    def rot_trans_data(self, bp_datapath):
        # FORMATING DATA
        time_length = len(self.df_data) # t
        self.data = np.swapaxes(self.df_data.reshape((time_length,30,3)).T,0,1) # 30 x 3 x t
        # MAKE POINT 2 ORIGIN
        self.trans_data = np.copy(self.data[2,0:2,:])
        self.data[:,0:2,:] = self.data[:,0:2,:] - self.trans_data
        # COMPUTE THE CENTER OF AXIS & ANGLE
        axis_vector = self.data[1,:,:]
        self.rot_data = np.sign(axis_vector[0,:])*np.pi/2-np.arctan(axis_vector[1,:]/axis_vector[0,:])
        # ROTATE ALL DATA POINT
        for i in range(30):
            self.data[i,0:2,:] = self.Rotate(self.data[i,0:2,:], self.rot_data)
        # SAVE DATA
        np.save(bp_datapath+"/"+self.filename, self.data)
        np.save(bp_datapath+"/trans_"+self.filename, self.trans_data)
        np.save(bp_datapath+"/rot_"+self.filename, self.rot_data)
        pass
    def Rotate(self, data, angle):
        return np.einsum('ijk,jk ->ik', np.array([[np.cos(angle), -1*np.sin(angle)],
                                                 [np.sin(angle), np.cos(angle)]]), data)
    def morlet_transform(self, power_datapath, w_0=15, N_f=25, f_min=1, f_max=6, fr=25, filename="", data=None):
        # MORLET TRANSFORM ON BODY PARTS
        MT = MorletTrans(w_0=w_0, N_f=N_f, f_min=f_min, f_max=f_max, fr=fr)
        if data is not None:
            self.power_bp = MT.run(data[:,0:2,:]) 
        else:
            # try:
            #     self.power_bp = np.append(self.power_bp, MT.run(self.data[:,0:2,:]), axis=3)
            # except:
            #     self.power_bp = MT.run(self.data[:,0:2,:]) 
            self.power_bp = MT.run(self.data[:,0:2,:]) 
        # IF YOU WANT A PLOT
        # MT.plot(self.power_bp, save_as_pdf=False, t_start=0, t_end=-1)  
        if filename:
            np.save(power_datapath+"/"+filename, self.power_bp)
        else:
            np.save(power_datapath+"/"+self.filename, self.power_bp)
        pass
    
def main():
    # ALGORITHM TO CLASSIFY BEHAVIOR
    CB = Morlet()
    data_pathnames = glob.glob('data/20181005_PP_food1DeepCut_resnet50_AntJan3shuffle1_1030000.h5')

    bp_datapath = "data/bp_data"
    power_datapath = "data/pwr_data"

    for data_path in tqdm(data_pathnames):
        CB.import_data(data_path)
        print("FINISHED Importing Data")
        CB.rot_trans_data(bp_datapath)
        print("FINISHED Rotating & Translating Data")
        CB.morlet_transform(power_datapath)
        print("FINISHED Morlet Transform")
    
    # np.save(power_datapath, CB.power_bp[:,:,:,:])
    
if __name__ == "__main__":
    main()