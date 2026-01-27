from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class DataProcessor:
    def __init__(self, batch_size: int = 32, train: bool = True):
        self.batch_size = batch_size
        self.loader = None
        try:
            self.loader = self.load_cifar10_dataset(self.batch_size, train=train)
        except Exception as e:
            print(f"Caught exception while trying to load the torch dataset: {e}")

    def load_cifar10_dataset(
        self, batch_size: int = 32, train: bool = True
    ) -> DataLoader | None:

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if train:

            train_dataset = torchvision.datasets.CIFAR10(
                root="../data/cifar-10-python",
                train=True,
                transform=transform,
                download=True,
            )

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=True
            )

            if train_loader == None:
                raise ValueError("DataLoader is None")

            output_loader = train_loader

        else:
            test_dataset = torchvision.datasets.CIFAR10(
                root="../data/cifar-10-python",
                train=False,
                transform=transform,
                download=True,
            )

            test_loader = DataLoader(
                dataset=test_dataset, batch_size=self.batch_size, shuffle=True
            )

            if test_loader == None:
                raise ValueError("DataLoader is None")

            output_loader = test_loader

        return output_loader

    def get_loader(self) -> DataLoader | None:
        return self.loader


if __name__ == "__main__":
    batch_size = 8
    processor = DataProcessor(batch_size=batch_size, train=True)
