import numpy as np
# tool function
def translational(data, origin_bp):
    # data format: num_bp x num_coord x t
    # origin_bp - specifies which body point to make the origin
    return ( np.copy(data - data[origin_bp,:,:]), np.copy(data[origin_bp,:,:]) )

def rotational(data, axis_bp):
    # rotate axis to be vertical; only works with 2 dimensions as of right now
    # data format: num_bp x (X_coord, Y_coord) x t
    # angle_list: angle of rotation from the vertical per frame
    rot_data = np.copy(data)
    num_bp = rot_data.shape[0]
    axis_vector = rot_data[axis_bp,:,:]
    angle_list = np.sign(axis_vector[0,:]) * np.pi/2 - np.arctan( axis_vector[1,:]/axis_vector[0,:] ) # angle rotated per frame
    # rotate each body point
    for i in range(num_bp):
        rot_data[i,:,:] = Rotate(rot_data[i,:,:], angle_list)
    return (rot_data, angle_list)

# helper tool function
def Rotate(data, angle):
    return np.einsum('ijk,jk ->ik', np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]]), data)
