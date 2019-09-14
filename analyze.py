import os
import numpy as np
import scipy.io as sio
import pandas as pd

def collect_data(num_files=1):
    data = None
    count = 0
    for root, dirs, files in os.walk("./data_result/data_final"):
        # print("root: ", root); print("dirs: ", dirs); print("files: ", files)
        
        if "EMBED.mat" in files:
            data_i = sio.loadmat(root+"/EMBED.mat")['embed_values_i']
            df = pd.DataFrame(data_i, columns=['x','y'])
            print(df)


            print(":: file #:", count, "   root:", root)
            data = np.vstack((data, data_i)) if data is not None else data_i
            count += 1
        if count == num_files:
            break
    print(":: Number of files were loaded: ", count)
    print(":: Shape of Data: ", data.shape)
    return data
def ethogram():

    pass

if __name__ == "__main__":
    data = collect_data()