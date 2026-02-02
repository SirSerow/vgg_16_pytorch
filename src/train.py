import torch.optim as optim
import torch.nn as nn
import torch
import os
import datetime
import argparse
from loguru import logger
from tqdm import tqdm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


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


def train(
    device: str = "cpu",
    pretrained_weights: Optional[str] = None,
    batch_size: Optional[int] = 4,
    learning_rate: float = 0.001,
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    vgg_16 = VGG16(num_classes=10).to(device)

    if batch_size is None:
        batch_size = 4

    if pretrained_weights and os.path.isfile(pretrained_weights):
        vgg_16.load_state_dict(torch.load(pretrained_weights, map_location=device))
        logger.info(f"Loaded pretrained weights from {pretrained_weights}")
    else:
        logger.info("No pretrained weights provided, training from scratch.")

    train_processor = DataProcessor(batch_size=batch_size, train=True)

    train_loader = train_processor.get_loader()

    test_processor = DataProcessor(batch_size=batch_size, train=False)
    test_loader = test_processor.get_loader()

    learning_rate = learning_rate

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(vgg_16.parameters(), lr=learning_rate)

    num_epochs = 10

    if train_loader is not None and test_loader is not None:

        logger.info(f"Loaded train dataset of {len(train_loader)} elements")
        logger.info(f"Loaded test dataset of {len(test_loader)} elements")

        date_started = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        output_folder = create_output_folder(f"vgg16_{date_started}")

        writer = SummaryWriter(output_folder)

        global_step = 0

        for epoch in range(num_epochs):

            vgg_16.train()

            for images, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                ncols=80,
                dynamic_ncols=True,
                leave=False,
                unit="batch",
                mininterval=0.5,
            ):

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = vgg_16(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1

            # save weights every epoch
            torch.save(
                vgg_16.state_dict(),
                os.path.join(output_folder, f"vgg16_epoch_{epoch+1}.pth"),
            )

            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            vgg_16.eval()

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in tqdm(
                    test_loader,
                    desc="Evaluating",
                    ncols=80,
                    dynamic_ncols=True,
                    leave=False,
                    unit="batch",
                    mininterval=0.5,
                ):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = vgg_16(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                writer.add_scalar("Accuracy/test", accuracy, epoch)
                logger.info(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

        writer.close()

    else:
        logger.error("Train loader or test loader is None, cannot train the model.")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train VGG16 model")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or cuda)",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="",
        help="Path to the model weights file to continue training from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and testing",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    args = parser.parse_args()

    train(
        device=args.device,
        pretrained_weights=args.weight_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Example command to run the training script:
    # python src/train.py --device cuda --weight_path ../.output/train/vgg16_240623_153045/vgg16_epoch_5.pth --batch_size 8 --learning_rate 0.0001
