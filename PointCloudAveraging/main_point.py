"""
Gradient flows in 2D
====================

Let's showcase the properties of **kernel MMDs**, **Hausdorff**
and **Sinkhorn** divergences on a simple toy problem:
the registration of one blob onto another.
"""
import ot

##############################################
# Setup
# ---------------------
import random
import time
from utils import *
import numpy as np
import torch


torch.manual_seed(1)
np.random.seed(1)
A = np.load("reconstruct_random_50_shapenetcore55.npy")
ind1=6#  16 18 6 46
ind2=46
target2=A[ind2]
target1=A[ind1]
np.save("saved/target{}.npy".format(ind1),target1)
np.save("saved/target{}.npy".format(ind2),target2)
device='cuda'
learning_rate = 0.01
N_step=1000
eps=0
L=10
print_steps = [0,9,199,499,999,1499,1999,2999,3999,4999,5999,6999,7999,8999,9999]
X1 = torch.from_numpy(target1).to(device)
X2 = torch.from_numpy(target2).to(device)
source = torch.randn(X1.shape)
source= source/torch.sqrt(torch.sum(source**2,dim=1,keepdim=True))
N=target1.shape[0]
copy=False
Ls=[10]
seeds=[1,2,3]
mars= torch.stack([X1,X2],dim=0)

lam=0.1
torch.manual_seed(1)
np.random.seed(1)
for L in Ls:
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
                distance1,distance2 = np.abs(distance1-distance2),0.5*(distance1+distance2)
                print("W {}:{}, {} ({}s)".format(i + 1,distance1,distance2 ,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append([distance1,distance2])
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # x= X.detach().numpy()
                # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
                # plt.show()
                np.save("saved/OBSW_{}_L{}_lam{}_{}_{}_points_seed{}.npy".format(i, L,lam, ind1, ind2, seed),
                        X.clone().cpu().data.numpy())
            optimizer.zero_grad()
            sw= OBSW(mars,X,L=L,lam=lam,device=device)
            loss= N*sw
            loss.backward()
            optimizer.step()

        np.savetxt("saved/OBSW_L{}_lam{}_{}_{}_distances_seed{}.txt".format(L,lam,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/OBSW_L{}_lam{}_{}_{}_times_seed{}.txt".format(L,lam,ind1,ind2,seed), np.array(caltimes), delimiter=",")

# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 distance1,distance2 = np.abs(distance1-distance2),0.5*(distance1+distance2)
#                 print("W {}:{}:{}, {} ({}s)".format(i + 1,distance1,distance2,distance1+distance2 ,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#                 np.save("saved/BSW_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#             optimizer.zero_grad()
#             sw= BSW(mars,X,L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#
#         np.savetxt("saved/BSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/BSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")
#
# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 distance1, distance2 = np.abs(distance1 - distance2), 0.5 * (distance1 + distance2)
#                 print("W {}:{}:{}, {} ({}s)".format(i + 1,distance1,distance2,distance1+distance2 ,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/FBSW_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw = FBSW(mars, X, L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/FBSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/FBSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")
#
#
# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 distance1, distance2 = np.abs(distance1 - distance2), 0.5 * (distance1 + distance2)
#                 print("W {}:{}:{}, {} ({}s)".format(i + 1,distance1,distance2,distance1+distance2 ,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/FBSWl_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw= lowerboundFBSW(mars,X,L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/FESWl_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/FESWl_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")
#
# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 distance1, distance2 = np.abs(distance1 - distance2), 0.5 * (distance1 + distance2)
#                 print("W {}:{}:{}, {} ({}s)".format(i + 1,distance1,distance2,distance1+distance2 ,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/EFBSW_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw = EFBSW(mars, X, L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/EFBSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/EFBSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")


# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 distance1, distance2 = np.abs(distance1 - distance2), 0.5 * (distance1 + distance2)
#                 print("W {}:{}, {} ({}s)".format(i + 1, distance1, distance2, np.round(cal_time, 2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/EFBSWl_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw= lowerbound_EFBSW(mars,X,L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/EFBSWl_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/EFBSWl_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")



# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 print("W {}:{}, {} ({}s)".format(i + 1, np.abs(distance1-distance2),distance1+distance2,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/FSW_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw = FEFBSW(mars, X, L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/FEFSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/FEFSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")
#
#
# torch.manual_seed(1)
# np.random.seed(1)
# for L in Ls:
#     for seed in seeds:
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         X=torch.tensor(source, requires_grad=True,device=device)
#         optimizer = torch.optim.SGD([X], lr=learning_rate)
#         points=[]
#         caltimes=[]
#         distances=[]
#         start = time.time()
#         for i in range(N_step):
#             if (i in print_steps):
#                 distance1,distance2,cal_time=compute_true_Wasserstein(X, X1),compute_true_Wasserstein(X, X2), time.time() - start
#                 print("W {}:{}, {} ({}s)".format(i + 1, np.abs(distance1-distance2),distance1+distance2,np.round(cal_time,2)))
#                 points.append(X.clone().cpu().data.numpy())
#                 caltimes.append(cal_time)
#                 distances.append([distance1,distance2])
#                 np.save("saved/FESWmm_{}_L{}_{}_{}_points_seed{}.npy".format(i, L, ind1, ind2, seed),
#                         X.clone().cpu().data.numpy())
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(projection='3d')
#                 # x= X.detach().numpy()
#                 # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
#                 # plt.show()
#
#             optimizer.zero_grad()
#             sw= lowerbound_FEFBSW(mars,X,L=L,device=device)
#             loss= N*sw
#             loss.backward()
#             optimizer.step()
#         points.append(X.clone().cpu().data.numpy())
#         # np.save("saved/FSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
#         np.savetxt("saved/FEFESWl_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
#         np.savetxt("saved/FEFESWl_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")