import torch.optim as optim
import torch.nn as nn
import torch
import os
import datetime

from utils.dataloader import DataProcessor
from utils.model import VGG16


def create_output_folder(name: str) -> str:
    # Check if ../.output folder exists, if not create it
    output_path = os.path.join("..", ".output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # check if folder ../.output/train exists, if not create it
    train_path = os.path.join(output_path, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    final_path = os.path.join(train_path, name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    return final_path


def train():

    vgg_16 = VGG16(num_classes=10)

    batch_size = 4

    processor = DataProcessor(batch_size=batch_size)

    train_loader = processor.get_train_loader()

    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg_16.parameters(), lr=learning_rate)

    num_epochs = 10

    if train_loader is not None:

        print(f"Loaded train dataset of {len(train_loader)} elements")

        date_started = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        output_folder = create_output_folder(f"vgg16_{date_started}")

        for epoch in range(num_epochs):

            progress_counter = 0

            for images, labels in train_loader:

                current_progress = 0
                # Forward pass
                outputs = vgg_16(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_counter += 1
                current_progress = (
                    (progress_counter * batch_size) / len(train_loader)
                ) * 100

                print(f"Current progress: {current_progress} %")

            # save weights every epoch
            torch.save(
                vgg_16.state_dict(),
                os.path.join(output_folder, f"vgg16_epoch_{epoch+1}.pth"),
            )

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    else:
        print("Train loader is None, cannot train the model.")


if __name__ == "__main__":
    train()
