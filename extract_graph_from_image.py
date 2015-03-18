# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

import scipy.ndimage
import networkx as nx

N_REGIONS = 16

# Craig and Shandice:
# These modifications are just to get something working. Both changes (flattening
# to a grayscale representation and taking the entries to be integers) were just
# to get apple as close to sp.misc.lena() as needed for the script to work.

def load_image(path):
    image_bw_big = (sp.misc.imread(path, flatten=True)).astype(np.int64)
    image_c_big = (sp.misc.imread(path)).astype(np.int64)

    # Downsample the image by a factor of 4
    # The color image can just use scipy.misc.imresize, but not the black and white one
    # if we resize the image it will map each block of 16 pixels to a pixel in the range 0-255 (2^0 -1 to 2^8 -1)
    # this method puts each pixel as 0-4096 (2^0 -1 to 2^12 -1)
    imagebw = image_bw_big
    imagebw = imagebw[::2, ::2] + imagebw[1::2, ::2] + imagebw[::2, 1::2] + imagebw[1::2, 1::2]
    imagebw = imagebw[::2, ::2] + imagebw[1::2, ::2] + imagebw[::2, 1::2] + imagebw[1::2, 1::2]
    #imagebw = scipy.misc.imresize(image_bw_big,.25)
    imagec = scipy.misc.imresize(image_c_big,.25)
    return imagebw, imagec

def cluster(imagebw):
    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(imagebw)

    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    beta = 5
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / imagebw.std()) + eps

    # Apply spectral clustering (this step goes much faster if you have pyamg
    # installed)
    #for assign_labels in ('kmeans'): #, 'discretize'
    assign_labels = 'discretize'
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels,
                                 random_state=1)
    t1 = time.time()
    t = t1 - t0
    labels = labels.reshape(imagebw.shape)
    return labels, t

def create_connectivity_matrix(labels):
    edges = dict()
    for l in range(N_REGIONS):
        edges[l] = scipy.ndimage.morphology.binary_dilation(labels == l)
    connectivity = np.zeros((N_REGIONS,N_REGIONS))
    for i in range(N_REGIONS):
        for j in range(N_REGIONS):
            connectivity[i,j] = np.count_nonzero(edges[i] & edges[j]) > 0
    return connectivity
    
def find_centroids(labels):
    centroids = np.zeros((N_REGIONS,2))
    for i in range(N_REGIONS):
        centroids[i,:] = scipy.ndimage.measurements.center_of_mass(labels == i)
    return centroids

def find_colors(labels, imagec):
    colors = np.zeros((N_REGIONS,3))
    for i in range(N_REGIONS):
        colors[i,:] = np.mean(imagec[labels == i],axis=0)
    return colors
    
def find_counts(labels):
    counts = np.zeros((N_REGIONS,1))
    for i in range(N_REGIONS):
        counts[i] = np.sum(labels == i)
    return counts


def show_plot(imagebw, labels, t):
    plt.figure(figsize=(5, 5))
    plt.imshow(imagebw,   cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title('Spectral clustering:  %.2fs' % (t))
    plt.show()


def show_graph(labels, imagec):
    plt.figure(figsize=(8,8))
    connectivity = create_connectivity_matrix(labels)
    G = nx.from_numpy_matrix(connectivity)
    centroids = find_centroids(labels)
    colors = find_colors(labels, imagec) / 255
    size = find_counts(labels)
    nx.draw(G,  pos=centroids, node_color=colors, node_size=size)
    plt.show()

if __name__ == "__main__":
    imagebw, imagec = load_image("apple.jpg")
    labels, t = cluster(imagebw)
    #show_plot(imagebw, labels, t)
    show_graph(labels, imagec)
    