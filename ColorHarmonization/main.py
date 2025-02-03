import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
from sklearn import cluster
import torch
import time
import argparse
from utils import *
import ot
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
# s6.bmp, t4.bmp, t6.bmp  $ fleur-3.jpg, fleur-1.jpg, fleur-2.jpg
parser = argparse.ArgumentParser(description='CT')
parser.add_argument('--num_iter', type=int, default=5000, metavar='N',
                    help='Num Interations')
parser.add_argument('--lam', type=int, default=1, metavar='N',
                    help='Num Interations')
parser.add_argument('--source', default="images/s6.bmp", type=str, metavar='N',
                    help='Source')
parser.add_argument('--target1', default="images/t4.bmp",type=str, metavar='N',
                    help='Target')
parser.add_argument('--target2', default="images/t6.bmp",type=str, metavar='N',
                    help='Target')


args = parser.parse_args()


n_clusters = 3000
name1=args.source#path to images 1
name2=args.target1#path to images 2
name3=args.target2#path to images 2
source = img_as_ubyte(io.imread(name1))
target1 = img_as_ubyte(io.imread(name2))
target2 = img_as_ubyte(io.imread(name3))
reshaped_target1 = img_as_ubyte(resize(target1, source.shape[:2]))
reshaped_target2 = img_as_ubyte(resize(target2, source.shape[:2]))
name1=name1.replace('/', '')
name2=name2.replace('/', '')
name3=name3.replace('/', '')

os.makedirs('npzfiles', exist_ok=True)

X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
source_k_means.fit(X)
source_values = source_k_means.cluster_centers_.squeeze()
source_labels = source_k_means.labels_

# create an array from labels and values
source_compressed = source_values[source_labels]
source_compressed.shape = source.shape

vmin = source.min()
vmax = source.max()




X = target1.reshape((-1, 3))  # We need an (n_sample, n_feature) array
target1_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
target1_k_means.fit(X)
target1_values = target1_k_means.cluster_centers_.squeeze()
target1_labels = target1_k_means.labels_

# create an array from labels and values
target1_compressed = target1_values[target1_labels]
target1_compressed.shape = target1.shape


X = target2.reshape((-1, 3))  # We need an (n_sample, n_feature) array
target2_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
target2_k_means.fit(X)
target2_values = target2_k_means.cluster_centers_.squeeze()
target2_labels = target2_k_means.labels_

# create an array from labels and values
target2_compressed = target2_values[target2_labels]
target2_compressed.shape = target2.shape


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
start = time.time()
BSWcluster,BSW = transform(source_values,target1_values,target2_values,source_labels,source,sw_type='bsw',num_iter=args.num_iter)
BSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
start = time.time()
OBSWcluster,OBSW = transform(source_values,target1_values,target2_values,source_labels,source,lam=args.lam,sw_type='obsw',num_iter=args.num_iter)
OBSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
start = time.time()
FBSWcluster,FBSW = transform(source_values,target1_values,target2_values,source_labels,source,sw_type='fbsw',num_iter=args.num_iter)
FBSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
start = time.time()
FBSWlcluster,FBSWl = transform(source_values,target1_values,target2_values,source_labels,source,sw_type='fbswl',num_iter=args.num_iter)
FBSWltime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
start = time.time()
EFBSWcluster,EFBSW = transform(source_values,target1_values,target2_values,source_labels,source,sw_type='efbsw',num_iter=args.num_iter)
EFBSWtime = np.round(time.time() - start,2)


source3=source_values.reshape(-1,3)
reshaped_target31,reshaped_target32=target1_values.reshape(-1,3),target2_values.reshape(-1,3)



# f.suptitle("L={}, k={}, T={}".format(L, k, iter), fontsize=20)
C_BSW1,C_BSW2 = ot.dist(BSWcluster,reshaped_target31),ot.dist(BSWcluster,reshaped_target32)
C_OBSW1,C_OBSW2 = ot.dist(OBSWcluster,reshaped_target31),ot.dist(OBSWcluster,reshaped_target32)
C_FBSW1,C_FBSW2 = ot.dist(FBSWcluster,reshaped_target31),ot.dist(FBSWcluster,reshaped_target32)
C_FBSWl1,C_FBSWl2 = ot.dist(FBSWlcluster,reshaped_target31),ot.dist(FBSWlcluster,reshaped_target32)
C_EFBSW1,C_EFBSW2 = ot.dist(EFBSWcluster,reshaped_target31),ot.dist(EFBSWcluster,reshaped_target32)


W_BSW = np.round(np.abs(ot.emd2([],[],C_BSW1) - ot.emd2([],[],C_BSW2))*1,3 )
W_OBSW = np.round(np.abs(ot.emd2([],[],C_OBSW1) - ot.emd2([],[],C_OBSW2))*1,3 )
W_FBSW = np.round(np.abs(ot.emd2([],[],C_FBSW1) - ot.emd2([],[],C_FBSW2))*1,3 )
W_FBSWl = np.round(np.abs(ot.emd2([],[],C_FBSWl1) - ot.emd2([],[],C_FBSWl2))*1,3 )
W_EFBSW = np.round(np.abs(ot.emd2([],[],C_EFBSW1) - ot.emd2([],[],C_EFBSW2))*1,3 )
# W_EFBSWl = np.round(np.abs(ot.emd2([],[],C_EFBSWl1) - ot.emd2([],[],C_EFBSWl2))*1,3 )


W_BSW2 = np.round(np.abs(ot.emd2([],[],C_BSW1) + ot.emd2([],[],C_BSW2))*0.5,3 )
W_OBSW2 = np.round(np.abs(ot.emd2([],[],C_OBSW1) + ot.emd2([],[],C_OBSW2))*0.5,3 )
W_FBSW2 = np.round(np.abs(ot.emd2([],[],C_FBSW1) + ot.emd2([],[],C_FBSW2))*0.5,3 )
W_FBSWl2 = np.round(np.abs(ot.emd2([],[],C_FBSWl1) + ot.emd2([],[],C_FBSWl2))*0.5,3 )
W_EFBSW2 = np.round(np.abs(ot.emd2([],[],C_EFBSW1) + ot.emd2([],[],C_EFBSW2))*0.5,3 )




f, ax = plt.subplots(1, 5, figsize=(12, 5))
ax[0].set_title('USWB $F={},W={}$'.format(W_BSW,W_BSW2), fontsize=12)
ax[0].imshow(BSW)

ax[1].set_title('MFSWB $\lambda={},F={},W={}$'.format(args.lam,W_OBSW,W_OBSW2), fontsize=12)
ax[1].imshow(OBSW)
ax[2].set_title('s-MFSWB  $F={},W={}$'.format(W_FBSWl,W_FBSWl2), fontsize=12)
ax[2].imshow(FBSWl)

ax[3].set_title('us-MFSWB $F={},W={}$'.format(W_FBSW,W_FBSW2), fontsize=12)
ax[3].imshow(FBSW)

ax[4].set_title('es-MFSWB $F={},W={}$'.format(W_EFBSW,W_EFBSW2), fontsize=12)
ax[4].imshow(EFBSW)




for i in range(5):
        ax[i].get_yaxis().set_visible(False)
        ax[i].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
plt.show()




