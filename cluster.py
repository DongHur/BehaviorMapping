import numpy as np
import time
import scipy.io as sio
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
# from debacl import geom_tree as gtree
# import debacl 
import hdbscan
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors  as colors
import timeit
from collections import Counter

def collect_data(num_files=5):
    data = None
    count = 0
    for root, dirs, files in os.walk("./data_result/data_final"):
        # print("root: ", root); print("dirs: ", dirs); print("files: ", files)
        
        if "EMBED.mat" in files:
            # print("file #: ", count)
            # print("root: ", root)
            data_i = sio.loadmat(root+"/EMBED.mat")['embed_values_i']
            data = np.vstack((data, data_i)) if data is not None else data_i
            count += 1
        if count == num_files:
            break
    print(":: Number of files were loaded: ", count)
    print(":: Shape of Data: ", data.shape)
    return data
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
def sklearn_optics_clustering(data, mode="all", select_idx=None, metric='euclidean'):
    # metric: [cityblock, cosine, euclidean, l1, l2, manhattan]
    optics = OPTICS(max_eps=100, min_samples=5, xi=0.03, algorithm="kd_tree", metric=metric).fit(data)
    c = Counter(optics.labels_)
    num_cluster = len(c)
    print(c.items())
    print("Number of Clusters: ", num_cluster)
    print(optics.cluster_hierarchy_)
    # dendrogram plot
    # reach_dist = optics.reachability_[optics.ordering_]
    # max_reachability = np.max(reach_dist[np.isreal(reach_dist)])
    # reach_dist[np.isinf(reach_dist)] = max_reachability*1.5
    # plt.plot(np.arange(len(reach_dist)), reach_dist, alpha=0.3)

    # ax = plt.subplot(1,1,1)
    # space = np.arange(len(reach_dist))
    # for klass in range(0, num_cluster):
    #     Xk = space[optics.labels_ == klass]
    #     Rk = reach_dist[optics.labels_ == klass]
    #     ax.scatter(Xk, Rk, alpha=0.3)
    # # scatter and label plot
    if mode=="all":
        plt.subplot(131)
        ax1=plt.subplot(1, 3, 1)
        ax2=plt.subplot(1, 3, 2)
        ax3=plt.subplot(1, 3, 3)

        ax1.scatter(data[:,0], data[:,1], c=optics.labels_, s=1)
        ax1.set_xlim(-150,150)
        ax1.set_ylim(-150,150)

        for klass in range(0, num_cluster):
            datak = data[optics.labels_ == klass]
            ax2.plot(datak[:, 0], datak[:, 1], alpha=0.3)
        ax2.set_xlim(-150,150)
        ax2.set_ylim(-150,150)

        outlier_idx = np.where(optics.labels_ == -1)
        ax3.scatter(data[outlier_idx,0], data[outlier_idx,1], s=1)
        ax3.set_xlim(-150,150)
        ax3.set_ylim(-150,150)
    if mode=="core":
        ax = plt.subplot()
        for klass in range(0, num_cluster):
            datak = data[optics.labels_ == klass]
            ax.plot(datak[:, 0], datak[:, 1], alpha=0.3)
    if mode=="outlier":
        outlier_idx = np.where(optics.labels_ == -1)
        plt.scatter(data[outlier_idx,0], data[outlier_idx,1], s=1)
        plt.xlim(-150,150)
        plt.ylim(-150,150)
    if mode=="select" and select_idx!=None:
        outlier_idx = np.where(optics.labels_ == select_idx)
        plt.scatter(data[outlier_idx,0], data[outlier_idx,1], s=1)
        plt.xlim(-150,150)
        plt.ylim(-150,150)
    pass
def debacl_clustering(data):
    # (n,p) = data.shape
    # # Smoothing Parameter, Pruning Parameter
    # p_k, p_gamma = 0.005, 0.05
    # # Modify parameters based on number of data points
    # k, gamma = int(n*p_k), int(n*p_gamma)
    # # Perform clustering
    # print("here 1")
    # tree = debacl.construct_tree(data, k=k, prune_threshold=gamma)
    # # tree = gtree.geomTree(data, k, gamma, n_grid = None, verbose = False)
    # print(tree)
    # print("here 2")
    # # plot
    # fig = tree.plot()
    # print("here 3")
    # uc, nodes = tree.getClusterLabels(method = "all-mode")
    # print("here 4")
    # fig, ax = debacl.utils.plotForeground(data, uc, fg_alpha = 0.95, bg_alpha = 0.1, edge_alpha = 0.6, s = 60)
    # print("here 5")
    pass

def sklearn_kde(data, kernel="gaussian", center=None):
    xmin, xmax = np.min(data[:,0]), np.max(data[:,0])
    ymin, ymax = np.min(data[:,1]), np.max(data[:,1])
    X, Y = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
    position = np.vstack([X.flatten(), Y.flatten()]).T

    # determine kernel metadata; Takes awhile; do important sampling and run only on a subset; don't run everytime
    # grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20) # 20-fold cross-validation
    # grid.fit(data)
    # bandwidth = grid.best_params_['bandwidth']
    # print("bandwidth: ", bandwidth)
    
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
def hdbscan_clustering(data, min_cluster_size=10, plot_cluster=False, plot_cluster_noiseless=True, 
    plot_span_tree=False, plot_linkage_tree=False, plot_condense_tree=False):
    # data - [num_frames, num_dim]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=140, min_samples=30, allow_single_cluster=False, gen_min_span_tree=True)
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
    # format cluster color
    color_palette = sns.color_palette('hls', num_cluster)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)])
    # plot figures
    if plot_cluster:
        plt.figure("Labelled Scatter Plot")
        plt.scatter(data[:,0], data[:,1], s=1.5, c=cluster_member_colors, alpha=0.3)
    if plot_cluster_noiseless:
        plt.figure("Noiseless Labelled Scatter Plot")
        idx = clusterer.labels_ != -1
        plt.scatter(data[idx,0], data[idx,1], s=1.5, c=cluster_member_colors[idx], alpha=0.3)
    # plt.savefig(FIG_PATH+"Labelled Scatter Plot")
    return clusterer.labels_, clusterer.probabilities_ 

def gaussian_conv(data, k_nearest=5, num_points=120, plot_kernel=False, plot_hist=False, plot_conv=True):
    # data - [num_frames, num_dim]
    # knn computation
    nbrs = NearestNeighbors(n_neighbors=k_nearest+1, algorithm='kd_tree').fit(data)
    K_dist, K_idx = nbrs.kneighbors(data)
    K_matrix_idx = nbrs.kneighbors_graph(data).toarray()
    # gaussian conv computation
    sigma = np.median(K_dist[:,-1])
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
        plt.pcolormesh(X_H, Y_H, GH_conv.T)
        # plt.imshow(GH_conv, extent=[L_bound, U_bound, L_bound, U_bound])
        plt.title("Gaussian Kernel Convolution w/ Data Histogram")
        plt.xlabel("X1")
        plt.ylabel("X2")
    pass


if __name__ == "__main__":
    data = collect_data(num_files=-1)
    start_time = time.time()
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
        sklearn_optics_clustering(data, mode="all")
        print(":: Finished OPTICS: {}".format(round(time.time()-start_time, 2)))
    if False:
        debacl_clustering(data)
        print(":: Finished debacl: {}".format(round(time.time()-start_time, 2)))
    if True:
        hdbscan_clustering(data, min_cluster_size=29, 
            plot_span_tree=False, plot_linkage_tree=False, plot_condense_tree=False)
        print(":: Finished HDBSCAN: {}".format(round(time.time()-start_time, 2)))
    if True:
        gaussian_conv(data)
        print(":: Finished Gausian Convolution: {}".format(round(time.time()-start_time, 2)))
    plt.show()
