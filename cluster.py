import numpy as np
import time
import scipy.io as sio
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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
def optimal_hdbscan(data, num_mcs=20, num_ms=20):
    (num_data, num_dim) = data.shape
    mcs_list = np.linspace(10,110,num_mcs, dtype=int)
    ms_list = np.linspace(5,70,num_ms, dtype=int)
    # mcs_list = np.linspace(80,90,num_mcs, dtype=int)
    # ms_list = np.linspace(31,38,num_ms, dtype=int)
    XX, YY = np.meshgrid(mcs_list, ms_list)
    # compute hdbscan result
    BIC = np.zeros((num_mcs, num_ms))
    for mcs_i, mcs in enumerate(tqdm(mcs_list)):
        for ms_i, ms in enumerate(ms_list):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs), 
                min_samples=int(ms), 
                allow_single_cluster=False)
            clusterer.fit(data)
            # compute Bayesian Information Criterion
            prob = clusterer.probabilities_
            # prob[prob==0] = 0.1
            prob = prob[prob!=0]
            num_pts = len(prob)
            num_cluster = int(clusterer.labels_.max()+1)
            nat_L = np.log(prob).sum()
            print("********")
            print(nat_L)
            print(num_pts)
            print(num_cluster)
            print(num_dim)
            BIC[mcs_i, ms_i] = np.log(num_pts)*num_cluster*num_dim - 2*nat_L
    ind = np.unravel_index(np.argmin(data), data.shape)
    print("Optimal mcs: ", ind[0])
    print("Optimal ms: ", ind[1])
  
    # np.save("./num_cluster.npy", num_cluster)
    np.save("./BIC_2.npy", BIC)
    if True:
        # plt.figure("Optimal HDBSCAN Using BIC")
        # im = plt.pcolormesh(XX, YY, BIC.T, cmap="jet")
        # plt.title("BIC")
        # plt.xlabel("min_cluster_size")
        # plt.ylabel("min_samples")
        # fig.colorbar(im, ax=ax0)

        fig = go.Figure(data=go.Heatmap(z=BIC.T, x=mcs_list,y=ms_list))
        fig.show()
    pass

def hdbscan_clustering(data, min_cluster_size=10, min_samples=65, plot_cluster=False, plot_cluster_noiseless=True, 
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

def gmm_opt(data):
    n_list = np.arange(3,150)
    score_list = np.zeros(len(n_list))
    for idx, n_components in tqdm(enumerate(n_list)):
        gmm = GaussianMixture(n_components)
        gmm.fit(data)
        score_list[idx] = gmm.bic(data)
    np.save("GMM_BIC.npy", score_list)
    if True:
        plt.figure("GMM BIC")
        plt.xlabel("Number of Components")
        plt.ylabel("BIC (Bayesian Information Criterion)")
        plt.title("BIC")
        plt.plot(n_list, score_list)
    pass
def gmm_clustering(data, n_components=40):
    print(data.shape)
    gmm = GaussianMixture(n_components)
    gmm.fit(data)
    print(gmm.weights_)
    label = gmm.predict(data)
    print(label)
    print(np.max(gmm.predict_proba(data), axis=1).shape)
    # format cluster color
    color_palette = sns.color_palette('hls', n_components)
    cluster_colors = [color_palette[x] for x in label]
    # cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in zip(cluster_colors, np.max(gmm.predict_proba(data), axis=1))])
    cluster_member_colors = np.array([(x[0],x[1],x[2],p*0.1) for x, p in zip(cluster_colors, np.max(gmm.predict_proba(data), axis=1))])

    print(gmm.score(data))
    if True:
        plt.figure("GMM Clustering")
        plt.title("GMM Clustering")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.scatter(data[:,0],data[:,1], s=1.5, c=cluster_member_colors)
    pass
def bgmm_clustering(data, n_components=80, weight_concentration_prior=1e-2):
    bgmm = BayesianGaussianMixture(n_components=50, weight_concentration_prior=0.1)
    bgmm.fit(data)
    label = bgmm.predict(data)
    print("Number of Cluster: ", label.max())
    # print(bgmm.predict_proba(data))
    # format cluster color
    color_palette = sns.color_palette('hls', n_components)
    cluster_colors = [color_palette[x] for x in label]
    # cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)])
    if True:
        plt.figure("GMM Clustering")
        plt.title("GMM Clustering")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.scatter(data[:,0],data[:,1], s=1.5, c=cluster_colors)
    pass
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
        np.save(key+"/cluster.npy", label[idx:idx+value])
        idx += value
    pass

if __name__ == "__main__":
    # filepath = "./data"
    filepath = "./data_result/rat_test_data"
    data, file_info = collect_data(filepath)
    start_time = time.time()
    # *** HDBSCAN Parameter ***
    min_cluster_size, min_samples = 86, 32 # 86, 32 MOST OPTIMAL Number of Clusters:137 Points Classified: 51.23%
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
        gmm_opt(data)
        print(":: Finished GMM Optimization: {}".format(round(time.time()-start_time, 2)))
    if False:
        bgmm_clustering(data)
        # gmm_clustering(data)
        print(":: Finished GMM Clustering: {}".format(round(time.time()-start_time, 2)))
    if False:
        gaussian_conv(data)
        print(":: Finished Gausian Convolution: {}".format(round(time.time()-start_time, 2)))
    if False:
        save_clusters(cluster_label, cluster_prob, file_info)
        print(":: Finished Saving Cluster: {}".format(round(time.time()-start_time, 2)))
    plt.show()
