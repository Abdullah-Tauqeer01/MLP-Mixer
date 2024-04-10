import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        CIFAR10(root="./data", train=True, download=True)
        CIFAR10(root="./data", train=False, download=True)

    def setup(self, stage=None):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((self.args.img_size, self.args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.train_dataset = CIFAR10(root="./data", train=True, transform=transform_train)
        self.test_dataset = CIFAR10(root="./data", train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, num_workers=4, pin_memory=True)

def get_loader(args):
    data_module = CIFAR10DataModule(args)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    return train_loader, test_loader
