from pytorch_balanced_sampler import *
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


class BaseDataLoader:
    def __init__(self, data_dir="data/", train_batch_size=128, test_batch_size=64, num_classes=0):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.create_dataset()

    def create_dataset(self):
        pass

    def create_dataloader(self):
        instances_indices = torch.arange(len(self.train_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.train_dataset.targets) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(self.train_dataset.targets) // self.train_batch_size,
            alpha=0,
            kind='random'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader


class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir="data/", train_batch_size=250, test_batch_size=250):
        super(MNISTDataLoader, self).__init__(data_dir=data_dir,
                                              train_batch_size=train_batch_size,
                                              test_batch_size=test_batch_size,
                                              num_classes=10)

    def create_dataset(self):
        train_set = datasets.MNIST(root=self.data_dir,
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
        test_set = datasets.MNIST(root=self.data_dir,
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))
        self.train_dataset = train_set
        self.test_dataset = test_set
