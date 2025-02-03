import ot
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *
torch.manual_seed(1)
np.random.seed(1)
def evaluate(Xs,X):
    distances=[]
    K=Xs.shape[0]
    for i in range(K):
        distances.append(compute_true_Wasserstein(Xs[i],X))
    distances=np.array(distances)
    W=np.mean(distances)
    distances=distances.reshape(-1,1)
    M=ot.dist(distances,distances,p=1)
    F=np.sum(M)/(K*(K-1))
    return np.round(W,2),np.round(F,2)
mu_1 = np.array([0, 0])
cov_1 = np.array([[1, 0], [0, 1]])


mu_2 = np.array([20, 0])
cov_2 = np.array([[1, 0], [0, 1]])


mu_3 = np.array([18, 8])
cov_3 = np.array([[1, 0], [0, 1]])

mu_4 = np.array([18, -8])
cov_4 = np.array([[1, 0], [0, 1]])

X1 = ot.datasets.make_2D_samples_gauss(100, mu_1, cov_1)
X2 = ot.datasets.make_2D_samples_gauss(100, mu_2, cov_2)
X3 = ot.datasets.make_2D_samples_gauss(100, mu_3, cov_3)
X4 = ot.datasets.make_2D_samples_gauss(100, mu_4, cov_4)

plt.figure(figsize=(5,5))
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.legend()
plt.title('Marginals')
plt.tight_layout()
plt.show()


lr=0.01
T=5000
L=100
lam=1
Xs=np.stack([X1,X2,X3,X4],axis=0)
Xs = torch.from_numpy(Xs).float().cuda()
source = torch.randn((100,2))+torch.Tensor([0,-5])

plt.figure(figsize=(5,5))
torch.manual_seed(1)
np.random.seed(1)
X=torch.tensor(source, requires_grad=True,device='cuda')
optimizer = torch.optim.SGD([X], lr=lr)
for _ in range(T):
    optimizer.zero_grad()
    loss=OBSW(Xs,X,L=L,lam=lam,device='cuda')
    loss.backward()
    optimizer.step()
W,F=evaluate(Xs,X)
X=X.cpu().detach().numpy()
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.scatter(X[:,0],X[:,1],color='tab:cyan',marker='D',label=r'$\mu$')
plt.legend()
plt.title('MFSWB $\lambda={}$ F={}, W={}'.format(lam,F,W))
plt.tight_layout()
plt.show()


plt.figure(figsize=(5.3,5))
torch.manual_seed(1)
np.random.seed(1)
X=torch.tensor(source, requires_grad=True,device='cuda')
optimizer = torch.optim.SGD([X], lr=lr)
for _ in range(T):
    optimizer.zero_grad()
    loss=BSW(Xs,X,L=L,device='cuda')
    loss.backward()
    optimizer.step()
W,F=evaluate(Xs,X)
X=X.cpu().detach().numpy()
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.scatter(X[:,0],X[:,1],color='tab:cyan',marker='D',label=r'$\mu$')
plt.legend()
plt.ylabel('Iteration {}'.format(T),fontsize=14)
plt.title('USWB F={}, W={}'.format(F,W))
plt.tight_layout()
plt.show()
#
#
plt.figure(figsize=(5,5))
torch.manual_seed(1)
np.random.seed(1)
X=torch.tensor(source, requires_grad=True,device='cuda')
optimizer = torch.optim.SGD([X], lr=lr)
for _ in range(T):
    optimizer.zero_grad()
    loss=lowerboundFBSW(Xs,X,L=L,device='cuda')
    loss.backward()
    optimizer.step()
W,F=evaluate(Xs,X)
X=X.cpu().detach().numpy()
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.scatter(X[:,0],X[:,1],color='tab:cyan',marker='D',label=r'$\mu$')
plt.title('s-MFSWB F={}, W={}'.format(F,W))
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(5,5))
torch.manual_seed(1)
np.random.seed(1)
X=torch.tensor(source, requires_grad=True,device='cuda')
optimizer = torch.optim.SGD([X], lr=lr)
for _ in range(T):
    optimizer.zero_grad()
    loss=FBSW(Xs,X,L=L,device='cuda')
    loss.backward()
    optimizer.step()
W,F=evaluate(Xs,X)
X=X.cpu().detach().numpy()
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.scatter(X[:,0],X[:,1],color='tab:cyan',marker='D',label=r'$\mu$')
plt.title('us-MFSWB F={}, W={}'.format(F,W))
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,5))
torch.manual_seed(1)
np.random.seed(1)
X=torch.tensor(source, requires_grad=True,device='cuda')
optimizer = torch.optim.SGD([X], lr=lr)
for _ in range(T):
    optimizer.zero_grad()
    loss=EFBSW(Xs,X,L=L,device='cuda')
    loss.backward()
    optimizer.step()
W,F=evaluate(Xs,X)
X=X.cpu().detach().numpy()
plt.scatter(X1[:,0],X1[:,1],color='tab:red',marker='o',label=r'$\mu_1$')
plt.scatter(X2[:,0],X2[:,1],color='tab:green',marker='v',label=r'$\mu_2$')
plt.scatter(X3[:,0],X3[:,1],color='tab:olive',marker='s',label=r'$\mu_3$')
plt.scatter(X4[:,0],X4[:,1],color='tab:blue',marker='X',label=r'$\mu_4$')
plt.scatter(X[:,0],X[:,1],color='tab:cyan',marker='D',label=r'$\mu$')
plt.title('es-MFSWB F={}, W={}'.format(F,W))
plt.legend()
plt.tight_layout()
plt.show()