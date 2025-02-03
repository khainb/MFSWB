import torch
from metrics.wasserstein import *


class SWAEBatchTrainer:
    """ Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projections to approximate sliced metrics distance
            p (int): power of distance metric
            weight (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """

    def __init__(self, autoencoder, optimizer, distribution_fn, num_classes=10,
                 num_projections=200, p=2, weight_swd=1, weight_fsw=1, device=None, method="FEFBSW", lambda_obsw=1.):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self._device = device if device else torch.device('cpu')
        self.num_classes = num_classes

        self.weight = weight_swd
        self.weight_fsw = weight_fsw

        self.method = method
        self.lambda_obsw = lambda_obsw

    def train_on_batch(self, x, y):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x, y)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        with torch.no_grad():
            # reset gradients
            self.optimizer.zero_grad()
            # autoencoder forward pass and loss
            x = x.to(self._device)
            recon_x, z_posterior = self.model_(x)
            bce = F.binary_cross_entropy(recon_x, x)
            batch_size = x.size(0)
            z_prior = self._distribution_fn(batch_size).to(self._device)

            swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                              num_projections=self.num_projections_, p=self.p_,
                                              device=self._device)
            l1 = F.l1_loss(recon_x, x)

            loss = bce + float(self.weight) * swd + l1
            return {
                'loss': loss,
                'bce': bce,
                'w2': swd,
                'encode': z_posterior,
                'decode': recon_x,
                'l1': l1
            }

    def eval_on_batch(self, x, y):

        x = x.to(self._device)
        y = y.to(self._device)

        recon_x, z_posterior = self.model_(x)
        bce = F.binary_cross_entropy(recon_x, x)

        l1 = F.l1_loss(recon_x, x)

        batch_size = x.size(0)
        z_prior = self._distribution_fn(batch_size).to(self._device)

        swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                          num_projections=self.num_projections_, p=self.p_,
                                          device=self._device)

        list_z_posterior = list()
        for cls in range(self.num_classes):
            list_z_posterior.append(z_posterior[y == cls])

        if self.method == "FEFBSW":
            fsw = FEFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerbound_FEFBSW":
            fsw = lowerbound_FEFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "EFBSW":
            fsw = EFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerbound_EFBSW":
            fsw = lowerbound_EFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "FBSW":
            fsw = FBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)
        elif self.method == "lowerboundFBSW":
            fsw = lowerboundFBSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "BSW":
            fsw = BSW_list(Xs=list_z_posterior, X=z_prior, L=self.num_projections_, device=self._device)

        elif self.method == "OBSW":
            fsw = OBSW_list(Xs=list_z_posterior, X=z_posterior, L=self.num_projections_, lam=self.lambda_obsw, device=self._device)
        else:
            fsw = 0

        loss = bce + float(self.weight_fsw) * fsw + float(self.weight) * swd + l1

        return {
            'loss': loss,
            'bce': bce,
            'fsw_loss': fsw,
            'w2': swd,
            'encode': z_posterior,
            'decode': recon_x,
            'l1': l1
        }

    def forward(self, x):
        x = x.to(self._device)
        recon_x, z_posterior = self.model_(x)
        return {
            'encode': z_posterior,
            'decode': recon_x
        }
