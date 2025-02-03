from metrics.wasserstein import *
import matplotlib.pyplot as plt


def generate_image(model,
                   prior_distribution,
                   num_images=100,
                   device='cpu'):
    with torch.no_grad():
        z_sample = prior_distribution(num_images)
        model_device = next(model.parameters()).device
        x_synthesis = model.generate(z_sample.to(model_device)).to(device)
        return x_synthesis


def compute_fairness(list_metric):
    num_classes = len(list_metric)
    dist_pairwise = list()
    for cls_id_i in range(num_classes):
        for cls_id_j in range(cls_id_i + 1, num_classes):
            a = abs(list_metric[cls_id_i] - list_metric[cls_id_j])
            dist_pairwise.append(a)
    return torch.sum(torch.tensor(dist_pairwise)) / len(dist_pairwise)


def compute_averaging_distance(list_metric):
    return torch.sum(torch.tensor(list_metric)) / len(list_metric)


def convert_to_cpu_number(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        return x.item()
    else:
        return x


def plot_convergence(iterations, data, ylabel, title, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(iterations, data, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
def display_and_save_images(images, title=None, save_path=None, rows=5, cols=5):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10), facecolor='black')
    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            img = images[i * cols + j].cpu().numpy().squeeze()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='black')
    else:
        plt.show()