from metrics.wasserstein import *
from utils import generate_image, compute_fairness, compute_averaging_distance, convert_to_cpu_number


def compute_F_AD(list_features,
                 list_labels,
                 prior_distribution,
                 num_classes,
                 num_samples,
                 device):
    with torch.no_grad():
        dist_swd = list()
        for cls_id in range(num_classes):
            features_cls = list_features[list_labels == cls_id]
            z_samples = prior_distribution(num_samples).to(device)
            wd = compute_true_Wasserstein(X=features_cls, Y=z_samples)
            dist_swd.append(wd)
    return compute_fairness(dist_swd), compute_averaging_distance(dist_swd)


def ultimate_evaluation(args,
                        model,
                        test_loader,
                        prior_distribution,
                        device='cpu'):
    with torch.no_grad():
        model.eval()
        model.to("cpu")
        
        list_labels = list()
        list_encoded_images = list()
        list_real_images = list()
        RL = 0
        total_images = 0

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_real_images.append(x_test)
            list_labels.append(y_test)
            num_images = x_test.shape[0]
            total_images += num_images
            decoded_images, encoded_images = model(x_test)
            list_encoded_images.append(encoded_images.detach())
            RL += torch.nn.functional.binary_cross_entropy(x_test, decoded_images) * num_images

        tensor_real_images = torch.cat(list_real_images, dim=0)
        tensor_labels = torch.cat(list_labels, dim=0)
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0)

        print(f"Number of images in testing: {total_images}")
        # Compute RL
        RL = RL / total_images
        # Compute WG
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=total_images,
                                                 device="cpu")
        WG = compute_true_Wasserstein(X=tensor_generated_images.reshape(total_images, -1), Y=tensor_real_images.reshape(total_images, -1))
        # Compute LP
        prior_samples = prior_distribution(num_images)
        LP = compute_true_Wasserstein(X=tensor_encoded_images, Y=prior_samples)
        
        # Compute F and AD in latent space
        F, W = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             num_samples=total_images,
                             device="cpu")

        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        W = convert_to_cpu_number(W)
        
        model.to(device)
        return RL, LP, WG, F, W
