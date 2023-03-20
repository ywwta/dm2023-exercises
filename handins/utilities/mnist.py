import random
import scipy.io as sio
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
### END IMPORTS

from utilities.load_data import load_mnist
import csv
from itertools import count


def read_data_mnist():
    print("is in....")
    X_train, y_train, *_ = load_mnist()
    np.random.seed(133737)  # DO NOT CHANGE

    n, h, w = X_train.shape
    sel = np.random.permutation(n)
    X = X_train[sel]
    y = y_train[sel]
    # Load 200 random samples from the MNIST training set.
    X_reduced = X[:740].reshape(-1, 784)
    X_reduced = (X_reduced - X_reduced.mean(1, keepdims=True)) / X_reduced.std(1, keepdims=True)
    y_reduced = y[:740]
    indices = [0, 1, 3, 4, 7, 9, 16, 17, 18, 19, 21, 23, 24, 26, 28, 29, 30, 31, 33, 35, 36, 38, 41, 43, 47, 49, 51, 52, 53, 54, 55, 59, 69, 71, 72, 73, 74, 76, 79, 80, 81, 83, 84, 89, 90, 91, 97, 100, 101, 102, 103, 105, 106, 107, 111, 114, 119, 122, 123, 124, 126, 130, 132, 133, 135, 137, 139, 143, 144, 146, 150, 152, 155, 156, 158, 159, 160, 162, 163, 164, 165, 166, 168, 170, 171, 173, 176, 177, 179, 180, 182, 185, 188, 189, 191, 194, 197, 198, 199, 203, 204, 206, 209, 211, 213, 214, 216, 218, 221, 223, 224, 226, 229, 230, 232, 233, 234, 235, 243, 247, 249, 250, 251, 254, 257, 260, 261, 263, 264, 266, 267, 268, 269, 270, 271, 272, 273, 278, 279, 282, 283, 284, 285, 286, 287, 290, 292, 293, 296, 297, 300, 303, 304, 306, 307, 309, 312, 316, 317, 318, 320, 323, 325, 326, 327, 331, 332, 333, 334, 335, 338, 341, 342, 343, 344, 345, 346, 350, 353, 354, 357, 358, 359, 360, 362, 365, 369, 373, 374, 380, 381, 382, 383, 385, 388, 391, 392, 393, 394, 396, 398, 399, 403, 404, 406, 408, 413, 414, 415, 418, 419, 420, 421, 426, 427, 429, 434, 436, 437, 441, 443, 446, 449, 450, 451, 453, 455, 456, 457, 458, 459, 461, 462, 465, 467, 468, 476, 477, 480, 481, 483, 486, 490, 491, 498, 499, 500, 506, 507, 509, 510, 511, 513, 515, 516, 519, 521, 523, 524, 525, 529, 532, 533, 536, 538, 540, 548, 549, 552, 553, 555, 556, 559, 560, 562, 563, 566, 568, 569, 572, 574, 575, 578, 581, 582, 586, 590, 591, 592, 594, 596, 598, 600, 601, 604, 606, 607, 609, 610, 613, 617, 626, 629, 634, 636, 637, 638, 640, 641, 642, 644, 646, 647, 649, 653, 657, 662, 663, 666, 668, 672, 674, 675, 678, 680, 681, 683, 684, 685, 689, 691, 692, 694, 697, 698, 699, 701, 702, 703, 704, 707, 708, 709, 710, 711, 713, 714, 715, 718, 719, 720, 722, 723, 726, 727, 730, 731, 736]
  
    X_reduced = X_reduced[indices]
    y_reduced = y_reduced[indices]
    return X_reduced, y_reduced

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data_cifar():
    print("Reading cifar data...")
    X_train, y_train, *_ = load_mnist()
    np.random.seed(133737)  # DO NOT CHANGE
    k_dict = (unpickle('./cifar10/data_batch_1'))
    
    X = k_dict[b'data']
    y = k_dict[b'labels']
    
    
    
    n, dim = X.shape
    sel = np.random.permutation(n)
    X = X_train[sel]
    y = y_train[sel]
    # Load 200 random samples from the MNIST training set.
    X_reduced = X[:740]
    X_reduced = (X_reduced - X_reduced.mean(1, keepdims=True)) / X_reduced.std(1, keepdims=True)
    y_reduced = y[:740]
    return X_reduced, y_reduced


from sklearn.neighbors import NearestNeighbors
def plot_neighborhood_graph(G, pos, ax, y=[]):
    if len(y)==0:
        _, y_reduced = read_data_mnist()
    ax.axis('off')
    ax.set_aspect('equal')
    
    nx.draw_networkx_edges(G,pos=pos,ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G,pos=pos,ax=ax, node_color=y, node_size=50, cmap=plt.get_cmap('tab10'))
    
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    
def plot_img_neighborhood_graph(G, X, fig, pos, ax):
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    plt.figure(0, figsize=(10, 10))
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.3)
    plt.figure(1, figsize=(15, 15))
    
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    
    # Add images to graph
    piesize=0.02 # this is the image size
    p2=piesize/1.0
    for n in G.nodes:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) * np.array([1.001, 0.93]) + np.array([0., 0.04]) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        # a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(X[n].reshape(28, 28))
        a.axis('off')


