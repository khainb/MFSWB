import argparse
import os
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, rand, randn
from evaluate.eval import *
import torch.optim as optim
import torchvision.utils as vutils
from dataloader.dataloader import *
from utils import *
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')
    parser.add_argument('--method', type=str, default='FEFBSW', metavar='MED',
                        help='method (default: FEFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta2 (default: 0.999)')
    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (default: adam)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to log training status (default: 10)')
    parser.add_argument('--saved-model-interval', type=int, default=100, metavar='N',
                        help='number of epochs to save training artifacts (default: 1)')
    parser.add_argument('--lambda-obsw', type=float, default=1.0, metavar='OBSW',
                        help='hyper-parameter of OBSW method')
    args = parser.parse_args()
    # create output directory

    if args.method == "OBSW":
        args.method = f"OBSW_{args.lambda_obsw}"

    args.outdir = os.path.join(args.outdir, args.dataset)
    args.outdir = os.path.join(args.outdir, f"seed_{args.seed}")
    args.outdir = os.path.join(args.outdir, f"lr_{args.lr}")
    args.outdir = os.path.join(args.outdir, f"fsw_{args.weight_fsw}")
    args.outdir = os.path.join(args.outdir, args.method)

    args.datadir = os.path.join(args.datadir, args.dataset)

    outdir_checkpoint = os.path.join(args.outdir, "checkpoint")
    outdir_convergence = os.path.join(args.outdir, "convergence")
    outdir_latent = os.path.join(args.outdir, "latent")

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(outdir_checkpoint, exist_ok=True)
    os.makedirs(outdir_convergence, exist_ok=True)
    os.makedirs(outdir_latent, exist_ok=True)

    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"Method: {args.method}")
    if args.optimizer == 'rmsprop':
        print(
            'batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
                args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
            ))
    else:
        print(
            'batch size {}\nepochs {}\n{}: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
                args.batch_size, args.epochs, args.optimizer,
                args.lr, args.beta1, args.beta2, args.distribution,
                device.type, args.seed
            ))

    # build train and test set data loaders
    if args.dataset == 'mnist':
        data_loader = MNISTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = MNISTAutoencoder().to(device)
    else:
        raise NotImplementedError

    # create optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.dataset == 'mnist':
        if args.distribution == 'circle':
            distribution_fn = rand_cirlce2d
        elif args.distribution == 'ring':
            distribution_fn = rand_ring2d
        else:
            distribution_fn = rand_uniform2d
    else:
        raise ('distribution {} not supported'.format(args.distribution))

    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(autoencoder=model,
                               optimizer=optimizer,
                               distribution_fn=distribution_fn,
                               num_classes=data_loader.num_classes,
                               num_projections=args.num_projections,
                               weight_swd=args.weight_swd,
                               weight_fsw=args.weight_fsw,
                               device=device,
                               method=args.method,
                               lambda_obsw=args.lambda_obsw)

    list_loss = list()

    METHOD_NAME = {
        "EFBSW": "es-MFSWB",
        "FBSW": "us-MFSWB",
        "lowerboundFBSW": "s-MFSWB",
        "OBSW_0.1": "MFSWB $\lambda = 0.1$",
        "OBSW_1.0": "MFSWB $\lambda = 1.0$",
        "OBSW_10.0": "MFSWB $\lambda = 10.0$",
        "BSW": "USWB",
        "None": "SWAE"
    }

    for epoch in range(args.epochs):
        print('training...')
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader, start=0):
            batch = trainer.train_on_batch(x, y)

            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                    (batch_idx + 1), len(train_loader),
                    batch['loss'].item()))
        model.eval()

        with torch.no_grad():

            if (epoch + 1) % args.saved_model_interval == 0 or (epoch + 1) == args.epochs:

                # RL, LP, WG, F, W = ultimate_evaluation(args=args,
                #                                        model=model,
                #                                        test_loader=test_loader,
                #                                        prior_distribution=distribution_fn,
                #                                        device=device)
                # print(f"Evaluating method {args.method} at epoch: {epoch + 1}")
                # print(f" +) Reconstruction loss: {RL}")
                # print(f" +) Wasserstein distance between generated and real images: {WG}")
                # print(f" +) Wasserstein distance between posterior and prior distribution: {LP}")
                # print(f" +) Fairness: {F}")
                # print(f" +) Averaging distance: {W}")
                # print()
                
                test_encode, test_targets, test_loss = list(), list(), 0.0
                for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                    test_evals = trainer.test_on_batch(x_test)

                    test_encode.append(test_evals['encode'].detach())
                    test_loss += test_evals['loss'].item()
                    test_targets.append(y_test)
                test_loss /= len(test_loader)
                list_loss.append(test_loss)
                test_encode, test_targets = torch.cat(test_encode), torch.cat(test_targets)
                test_encode, test_targets = test_encode.cpu().numpy(), test_targets.cpu().numpy()
                print(f"Shape of test dataset to plot: {test_encode.shape}, {test_targets.shape}")

                # plot
                plt.figure(figsize=(10, 10))
                classes = np.unique(test_targets)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(classes)))
                for i, class_label in enumerate(classes):
                    plt.scatter(test_encode[test_targets == class_label, 0],
                                test_encode[test_targets == class_label, 1],
                                c=[colors[i]],
                                cmap=plt.cm.Spectral,
                                label=class_label)

                plt.legend()
                # plt.rc('text', usetex=True)
                # title = f'{METHOD_NAME[args.method]}' + " F={:.3f}, W={:.3f}".format(F, W)
                # title = f'{args.method}' + " F={:.3f}, W={:.3f}".format(F, W)
                title = f'{args.method}'
                plt.title(title)
                plt.savefig('{}/epoch_{}_test_latent.pdf'.format(outdir_latent, epoch))
                plt.close()

                e = epoch + 1
                outdir_end = os.path.join(outdir_checkpoint, f"epoch_{e}")
                imagesdir_epoch = os.path.join(outdir_end, "images")
                chkptdir_epoch = os.path.join(outdir_end, "model")

                os.makedirs(imagesdir_epoch, exist_ok=True)
                os.makedirs(chkptdir_epoch, exist_ok=True)
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(chkptdir_epoch, args.dataset, args.method))
                vutils.save_image(x, '{}/{}_train_samples.png'.format(imagesdir_epoch, args.distribution))
                vutils.save_image(batch['decode'].detach(),
                                  '{}/{}_train_recon.png'.format(imagesdir_epoch, args.distribution),
                                  normalize=True)
                gen_image = generate_image(model=model, prior_distribution=distribution_fn, num_images=500,
                                           device=device)
                vutils.save_image(gen_image,
                                  '{}/gen_image.png'.format(imagesdir_epoch), normalize=True)

    plot_convergence(range(1, len(list_loss) + 1), list_loss, 'Test loss',
                     f'In testing loss convergence plot of {args.method}',
                     f"{outdir_convergence}/test_loss_convergence.png")


if __name__ == '__main__':
    main()
