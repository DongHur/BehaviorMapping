import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def ant_bp_graph(data, frame=10):
    # plot ant points for specific time point t; specific to out setup with 30bp ants
    # data format: num_bp x (X_coord, Y_coord) x t
    plt.scatter(data[:,0,frame], data[:,1,frame])
    plt.plot(data[0:4,0,frame], data[0:4,1,frame])
    plt.plot(data[4:8,0,frame], data[4:8,1,frame])
    plt.plot(data[8:11,0,frame], data[8:11,1,frame])
    plt.plot(data[11:14,0,frame], data[11:14,1,frame])
    plt.plot(data[14:17,0,frame], data[14:17,1,frame])
    plt.plot(data[17:21,0,frame], data[17:21,1,frame])
    plt.plot(data[21:24,0,frame], data[21:24,1,frame])
    plt.plot(data[24:27,0,frame], data[24:27,1,frame])
    plt.plot(data[27:30,0,frame], data[27:30,1,frame])
    plt.xlim(left=-200, right=200)
    plt.ylim(bottom=-200, top=200)
    plt.gca().set_aspect('equal', 'box')
    plt.show()
    pass