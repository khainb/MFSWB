import numpy as np
import torch
import ot
from torch.nn.functional import pad
import torch.nn.functional as F


def quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    idx = torch.searchsorted(cws, qs, right=False).T
    return torch.gather(xs, 0, torch.clamp(idx, 0, n - 1))


def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)


def compute_Wasserstein(M, device='cpu', e=0):
    if (e == 0):
        pi = ot.emd([], [], M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(), reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi * M)


def rand_projections(dim, num_projections=1000, device='cpu'):
    projections = torch.randn((num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X, Y, theta, p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0, keepdim=True)
    return wasserstein_distance


def one_dimensional_Wasserstein(X, Y, theta, u_weights=None, v_weights=None, p=2):
    if (X.shape[0] == Y.shape[0] and u_weights is None and v_weights is None):
        return one_dimensional_Wasserstein_prod(X, Y, theta, p)
    u_values = torch.matmul(X, theta.transpose(0, 1))
    v_values = torch.matmul(Y, theta.transpose(0, 1))
    n = u_values.shape[0]
    m = v_values.shape[0]
    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                               dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                               dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    u_sorter = torch.sort(u_values, 0)[1]
    u_values = torch.gather(u_values, 0, u_sorter)

    v_sorter = torch.sort(v_values, 0)[1]
    v_values = torch.gather(v_values, 0, v_sorter)

    u_weights = torch.gather(u_weights, 0, u_sorter)
    v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    pad_width = [(1, 0)] + (qs.ndim - 1) * [(0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    qs = pad(qs, how_pad)

    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0)


def BSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    sw = torch.mean(wasserstein_distance, dim=1)
    return torch.mean(sw)


def BSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.mean(wasserstein_distance, dim=1)
    return torch.mean(sw)


def lowerboundFBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    sw = torch.mean(wasserstein_distance, dim=1)
    return torch.max(sw)


def lowerboundFBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.mean(wasserstein_distance, dim=1)

    return torch.max(sw)


def FBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    sw = torch.max(wasserstein_distance, dim=0)[0]
    return torch.mean(sw)


def FBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.max(wasserstein_distance, dim=0)[0]
    return torch.mean(sw)


def EFBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    sw = torch.max(wasserstein_distance, dim=0)[0]
    weight = torch.softmax(sw, dim=-1)
    return torch.sum(weight * sw)


def EFBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.max(wasserstein_distance, dim=0)[0]
    weight = torch.softmax(sw, dim=-1)
    return torch.sum(weight * sw)


def lowerbound_EFBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    sw = torch.max(wasserstein_distance, dim=0)[0]
    weight = torch.softmax(sw, dim=-1)
    return torch.max(torch.sum(weight.view(1, L) * wasserstein_distance, dim=1))


def lowerbound_EFBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.max(wasserstein_distance, dim=0)[0]
    weight = torch.softmax(sw, dim=-1)
    return torch.max(torch.sum(weight.view(1, L) * wasserstein_distance, dim=1))


def FEFBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    wasserstein_distanceT = wasserstein_distance.T.view(L, -1, 1)
    M = torch.cdist(wasserstein_distanceT, wasserstein_distanceT, p=2)

    weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    sws = torch.sum(weights * wasserstein_distance, dim=1)
    return torch.max(sws)


def FEFBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    wasserstein_distanceT = wasserstein_distance.T.view(L, -1, 1)
    M = torch.cdist(wasserstein_distanceT, wasserstein_distanceT, p=2)

    weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    return torch.sum(weights * torch.max(wasserstein_distance, dim=0)[0])


def lowerbound_FEFBSW(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod, dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted - X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)  # K\times L
    wasserstein_distanceT = wasserstein_distance.T.view(L, -1, 1)
    M = torch.cdist(wasserstein_distanceT, wasserstein_distanceT, p=2)

    weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    return torch.sum(weights * torch.max(wasserstein_distance, dim=0)[0])


def lowerbound_FEFBSW_list(Xs, X, L=10, p=2, device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta, p=p) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    wasserstein_distanceT = wasserstein_distance.T.view(L, -1, 1)
    M = torch.cdist(wasserstein_distanceT, wasserstein_distanceT, p=2)

    weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    sws = torch.sum(weights * wasserstein_distance, dim=1)
    return torch.max(sws)

def OBSW(Xs,X,L=10,lam=1,p=2,device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs = torch.cat(Xs, dim=0)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod,dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted-X_prod_sorted)
    sw = torch.mean(torch.pow(wasserstein_distance, p), dim=1)
    return torch.mean(sw)+lam*torch.cdist(sw.view(-1,1),sw.view(-1,1),p=1).sum()/(sw.shape[0]*sw.shape[0] - sw.shape[0])

def OBSW_list(Xs,X,L=10,lam=1,p=2,device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim=dim, num_projections=L, device=device)
    sw = [torch.mean(one_dimensional_Wasserstein(Xs[i], X, theta, p=p))for i in range(len(Xs))]
    sw = torch.tensor(sw)
    return torch.mean(sw) + lam * torch.cdist(sw.view(-1,1), sw.view(-1,1), p=1).sum() / (sw.shape[0]*sw.shape[0] - sw.shape[0])

def sliced_wasserstein_distance(encoded_samples,
                                distribution_samples,
                                num_projections=50,
                                p=2,
                                device='cpu',
                                theta=None):
    embedding_dim = distribution_samples.size(1)

    if theta is None:
        projections = rand_projections(dim=embedding_dim, num_projections=num_projections, device=device)
    else:
        projections = theta.to(device)
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()
