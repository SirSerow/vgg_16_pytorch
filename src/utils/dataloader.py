from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class DataProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        try:
            self.train_loader, self.test_loader = self.load_cifar10_dataset(
                self.batch_size
            )
        except Exception as e:
            print(f"Caught exception while trying to load the torch dataset: {e}")

    def load_cifar10_dataset(self, batch_size: int = 32):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="../data/cifar-10-python",
            train=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="../data/cifar-10-python",
            train=False,
            transform=transform,
        )

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=True
        )

        if test_loader == None or train_loader == None:
            raise ValueError("DataLoader is None")

        return train_loader, test_loader

    def get_train_loader(self) -> DataLoader | None:
        return self.train_loader

    def get_test_loader(self) -> DataLoader | None:
        return self.test_loader


if __name__ == "__main__":
    batch_size = 8
    processor = DataProcessor(batch_size=batch_size)
