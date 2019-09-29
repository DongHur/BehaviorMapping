import numpy as np
import time
import scipy.io as sio
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import hdbscan
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors  as colors
import timeit
from collections import Counter
from tqdm import tqdm

import plotly.graph_objects as go

def collect_data(filepath, num_files=-1):
    data = None
    count = 0
    file_info = {}
    for root, dirs, files in os.walk(filepath):
        # print("root: ", root); print("dirs: ", dirs); print("files: ", files)
        
        if "EMBED.mat" in files:
            # print("file #: ", count)
            # print("root: ", root)
            data_i = sio.loadmat(root+"/EMBED.mat")['embed_values_i']
            data = np.vstack((data, data_i)) if data is not None else data_i
            count += 1
            # save file information
            file_info[root] = data_i.shape[0]
        if count == num_files:
            break
    print(":: Number of files were loaded: ", count)
    print(":: Shape of Data: ", data.shape)
    return data, file_info
def sklearn_kmeans_clustering(data, start_n, stop_n):
    inertia = []
    for n_clusters in range(start_n, stop_n):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++").fit(data)
        inertia.append(kmeans.inertia_)
        # plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
        # plt.show()
    print("centers: ", kmeans.cluster_centers_)
    print("labels: ", kmeans.labels_)
    # Elbow Plot
    plt.plot(np.arange(start_n, stop_n), inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertial Value")
    plt.title("Elbow Plot")
    return np.array(kmeans.cluster_centers_), np.array(kmeans.labels_)
def sklearn_dbscan_clustering(data, mode="core", select_idx=None, metric='euclidean'):
    # metric: [cityblock, cosine, euclidean, l1, l2, manhattan]
    dbsc = DBSCAN(eps=2, min_samples=15, algorithm="kd_tree", metric=metric).fit(data)
    core_idx = dbsc.core_sample_indices_
    c = Counter(dbsc.labels_)
    num_cluster = len(c)
    print(c.items())
    print("Number of Clusters: ", num_cluster)
    if mode=="all":
        plt.scatter(data[:,0], data[:,1], c=dbsc.labels_, s=1)
    if mode=="core":
        plt.scatter(data[core_idx,0], data[core_idx,1], c=dbsc.labels_[core_idx], s=1, cmap="gist_rainbow")
    if mode=="outlier":
        outlier_idx = np.where(dbsc.labels_ == -1)
        plt.scatter(data[outlier_idx,0], data[outlier_idx,1], s=1)
        plt.xlim(-150,150)
        plt.ylim(-150,150)
    if mode=="select" and select_idx!=None:
        outlier_idx = np.where(dbsc.labels_ == select_idx)
        plt.scatter(data[outlier_idx,0], data[outlier_idx,1], s=1)
        plt.xlim(-150,150)
        plt.ylim(-150,150)
    pass

def sklearn_kde(data, kernel="gaussian", center=None):
    xmin, xmax = np.min(data[:,0]), np.max(data[:,0])
    ymin, ymax = np.min(data[:,1]), np.max(data[:,1])
    X, Y = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
    position = np.vstack([X.flatten(), Y.flatten()]).T
    
    # gaussian kernel
    bandwidth = 2
    kde_skl = KernelDensity(bandwidth=bandwidth, algorithm="kd_tree", kernel=kernel, metric="euclidean", 
        atol=0, rtol=1E-4)  
    kde_skl.fit(data)
    Z = kde_skl.score_samples(position)

    # Plot
    plt.pcolormesh(X, Y, Z.reshape(X.shape), cmap='jet')
    if center is not None:
        plt.scatter(center[:,0], center[:,1], marker="*")
    pass
def histogram(data, num_bins=50):
    plt.hist2d(data[:,0], data[:,1], bins=num_bins)
    pass

# https://towardsdatascience.com/lightning-talk-clustering-with-hdbscan-d47b83d1b03a
# https://umap-learn.readthedocs.io/en/latest/clustering.html
def optimal_hdbscan(data, num_mcs=50, num_ms=40):
    print(data.shape)
    mcs_list = np.linspace(10,300,num_mcs, dtype=int)
    ms_list = np.linspace(5,70,num_ms, dtype=int)
    XX, YY = np.meshgrid(mcs_list, ms_list)
    # compute hdbscan result
    num_cluster=np.zeros((num_mcs, num_ms))
    perc_label=np.zeros((num_mcs, num_ms))
    for mcs_i, mcs in tqdm(enumerate(mcs_list)):
        for ms_i, ms in enumerate(ms_list):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs), 
                min_samples=int(ms), 
                allow_single_cluster=False)
            clusterer.fit(data)
            num_cluster[mcs_i, ms_i] = int(clusterer.labels_.max()+1)
            perc_label[mcs_i, ms_i] = 100*len(np.where(clusterer.labels_!=-1)[0])/len(clusterer.labels_)
    print(num_cluster)
    print(perc_label)
    np.save("./num_cluster.npy", num_cluster)
    np.save("./perc_label.npy", perc_label)
    if True:
        plt.figure("Optimal HDBSCAN Plane Number of Cluster")
        plt.pcolormesh(XX, YY, num_cluster.T, cmap="jet")
        plt.title("Number of Cluster")
        plt.xlabel("min_cluster_size")
        plt.ylabel("min_samples")
    if True:
        plt.figure("Optimal HDBSCAN Plane Percent Labelled")
        plt.pcolormesh(XX, YY, perc_label.T, cmap="jet")
        plt.title("Percent Labelled")
        plt.xlabel("min_cluster_size")
        plt.ylabel("min_samples")
    pass

def hdbscan_clustering(data, min_cluster_size=140, min_samples=30, plot_cluster=False, plot_cluster_noiseless=True, 
    plot_span_tree=False, plot_linkage_tree=False, plot_condense_tree=False):
    # data - [num_frames, num_dim]
    if min_samples is None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            allow_single_cluster=False,
            gen_min_span_tree=True)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples, 
            allow_single_cluster=False,
            gen_min_span_tree=True)
    clusterer.fit(data)
    # plot figures
    if plot_span_tree:
        plt.figure("Minnimum Spanning Tree")
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
        # plt.savefig(FIG_PATH+"Minnimum Spanning Tree")
    if plot_linkage_tree:
        plt.figure("Linkage Tree")
        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # plt.savefig(FIG_PATH+"Linkage Tree")
    if plot_condense_tree:
        plt.figure("Condense Tree")
        clusterer.condensed_tree_.plot()
        # plt.savefig(FIG_PATH+"Condense Tree")
    # control density of color based on probability
    num_cluster = int(clusterer.labels_.max()+1)
    print("Number of Clusters: ", num_cluster)
    print("Points Classified: {}%".format(round(len(np.where(clusterer.labels_!=-1)[0])/len(clusterer.labels_)*100,2)))
    # format cluster color
    color_palette = sns.color_palette('hls', num_cluster)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)])
    # plot figures
    if plot_cluster:
        plt.figure("Labelled Scatter Plot")
        plt.title("Labelled Scatter Plot")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.scatter(data[:,0], data[:,1], s=1.5, c=cluster_member_colors)
    if plot_cluster_noiseless:
        plt.figure("Noiseless Labelled Scatter Plot")
        plt.title("Labelled Scatter Plot w/o Noise")
        plt.xlabel("X1")
        plt.ylabel("X2")
        idx = clusterer.labels_ != -1
        plt.scatter(data[idx,0], data[idx,1], s=1.5, c=cluster_member_colors[idx])

        fig = go.Figure(data=go.Scatter(x=data[idx,0], y=data[idx,1], 
            mode='markers', text=clusterer.labels_[idx], marker=dict(color=clusterer.labels_[idx], opacity=0.2)))
        fig.show()
    # plt.savefig(FIG_PATH+"Labelled Scatter Plot")
    return clusterer.labels_, clusterer.probabilities_ 

def gaussian_conv(data, k_nearest=5, num_points=250, plot_kernel=False, plot_hist=False, plot_conv=True):
    # data - [num_frames, num_dim]
    # knn computation
    nbrs = NearestNeighbors(n_neighbors=k_nearest+1, algorithm='kd_tree').fit(data)
    K_dist, K_idx = nbrs.kneighbors(data)
    K_matrix_idx = nbrs.kneighbors_graph(data).toarray()
    # gaussian conv computation
    sigma = np.median(K_dist[:,-1])*5.0
    print("sigma: ", sigma)
    L_bound = -1.0*abs(data.max())-1
    U_bound = 1.0*abs(data.max())+1
    xx = np.linspace(L_bound, U_bound, num_points)
    yy = np.linspace(L_bound, U_bound, num_points)
    XX, YY = np.meshgrid(xx, yy)
    # gaussian kernel
    G = np.exp(-0.5*(XX**2 + YY**2)/sigma**2)/(2*np.pi*sigma**2);
    if plot_kernel:
        plt.figure("Gaussian Kernel")
        plt.imshow(G, extent=[L_bound, U_bound, L_bound, U_bound])
        plt.title("Gaussian Kernel")
        plt.xlabel("X1")
        plt.ylabel("X2")
    # data histogram
    H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], num_points, [[L_bound,U_bound],[L_bound,U_bound]])
    X_H, Y_H = np.meshgrid(xedges, yedges)
    H = H/H.sum()
    if plot_hist:
        plt.figure("Data Histogram")
        plt.pcolormesh(X_H, Y_H, H.T)
        # plt.imshow(H, extent=[L_bound, U_bound, L_bound, U_bound]) 
        plt.title("Data Histogram")
        plt.xlabel("X1")
        plt.ylabel("X2")
    # fft convolution
    fr = np.fft.fft2(G)
    fr2 = np.fft.fft2(H)
    GH_conv = np.fft.fftshift(np.real(np.fft.ifft2(fr*fr2)))
    GH_conv[GH_conv<0] = 0
    if plot_conv:
        plt.figure("Gaussian Convolution")
        plt.pcolormesh(X_H, Y_H, GH_conv.T, cmap="jet")
        # plt.imshow(GH_conv, extent=[L_bound, U_bound, L_bound, U_bound])
        plt.title("Gaussian Kernel Convolution w/ Data Histogram")
        plt.xlabel("X1")
        plt.ylabel("X2")
    pass


def save_clusters(label, prob, file_info):
    idx = 0
    for key, value in file_info.items():
        # print(key)
        # print(value)
        np.save(key+"/cluster.npy", label[idx:idx+value])
        idx += value
    pass

if __name__ == "__main__":
    # filepath = "./data_result/sbatch_script1_25perc"
    filepath = "./data_result/data_final"
    data, file_info = collect_data(filepath)
    start_time = time.time()
    # *** HDBSCAN Parameter ***
    # old data
    # min_cluster_size, min_samples = 185, 28
    # Number of Clusters:  55
    # Points Classified: 79.77%
    # new data
    # min_cluster_size, min_samples = 185, 28
    # Number of Clusters:  55
    # Points Classified: 79.77%
    min_cluster_size, min_samples = 250, 28
    plot_cluster, plot_cluster_noiseless = False, True
    plot_linkage_tree, plot_condense_tree = False, False
    plot_span_tree = False
    
    
    if False:
        histogram(data, num_bins=100)
        print(":: Finished Histogram: {}".format(round(time.time()-start_time, 2)))
    if False:
        center, label = sklearn_kmeans_clustering(data, start_n=5, stop_n=20)
        sklearn_kde(data, kernel="exponential", center=center)
        print(":: Finished kmeans: {}".format(round(time.time()-start_time, 2)))
    if False:
        sklearn_dbscan_clustering(data, mode="core")
        # sklearn_dbscan_clustering(data, mode="select", select_idx=147)
        print(":: Finished DBSCAN: {}".format(round(time.time()-start_time, 2)))
    if False:
        debacl_clustering(data)
        print(":: Finished debacl: {}".format(round(time.time()-start_time, 2)))
    if False:
        optimal_hdbscan(data)
    if True:
        cluster_label, cluster_prob = hdbscan_clustering(data,  min_cluster_size, min_samples, plot_cluster, plot_cluster_noiseless,
            plot_span_tree, plot_linkage_tree, plot_condense_tree)
        print(":: Finished HDBSCAN: {}".format(round(time.time()-start_time, 2)))
    if False:
        gaussian_conv(data)
        print(":: Finished Gausian Convolution: {}".format(round(time.time()-start_time, 2)))
    if False:
        save_clusters(cluster_label, cluster_prob, file_info)
        print(":: Finished Saving Cluster: {}".format(round(time.time()-start_time, 2)))
    plt.show()
