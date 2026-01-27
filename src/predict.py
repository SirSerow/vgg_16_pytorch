import torch
import argparse
import os
from loguru import logger
import datetime
from tqdm import tqdm

from utils.dataloader import DataProcessor
from utils.model import VGG16


def predict(device: str = "cpu", weight_path: str = ""):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    vgg_16 = VGG16(num_classes=10).to(device)

    processor = DataProcessor(batch_size=1, train=False)

    test_loader = processor.get_loader()

    if test_loader is not None:
        logger.info(f"Loaded test dataset of {len(test_loader)} elements")

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found at {weight_path}")

        vgg_16.load_state_dict(torch.load(weight_path, map_location=device))

        vgg_16.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                test_loader,
                desc="Predicting",
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
        logger.info(f"Prediction Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Predict using VGG16 model")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for prediction (cpu or cuda)",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="Path to the model weights file",
    )
    args = parser.parse_args()

    predict(device=args.device, weight_path=args.weight_path)
