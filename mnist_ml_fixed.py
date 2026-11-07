"""
MNIST Minimal Machine Learning Practice
=======================================

Goal
----
Use PyTorch to complete a minimal MNIST 0~9 classification task.

IMPORTANT:
- You do NOT need to write everything from memory.
- You ARE encouraged to use ChatGPT / Codex / other AI Agents.
- However, when presenting your work, you should be able to explain
  what each BLOCK is doing.

What you should understand:
1. Dataset
2. DataLoader / Batch
3. Model
4. Loss Function
5. Optimizer / Learning Rate
6. Training Loop
7. Forward -> Loss -> Backward -> Update
8. Epoch
9. Inference
10. Accuracy

----------------------------------------------------------------------
BLOCK DIAGRAM
----------------------------------------------------------------------

MNIST Dataset
      |
      v
DataLoader
(batch)
      |
      v
Model
      |
      v
Forward
      |
      v
Prediction
      |
      v
Loss
      |
      v
Backward
(gradient)
      |
      v
Optimizer
(update parameters)
      |
      +---------- repeat for several epochs ----------+
                                                      |
                                                      v
                                                Trained Model
                                                      |
                                                      v
                                                  Inference
                                                      |
                                                      v
                                                   Accuracy

----------------------------------------------------------------------
Suggested way to ask an AI Agent
----------------------------------------------------------------------

Example prompt:

"I am a beginner learning PyTorch. I already have MNIST downloaded.
Please help me complete the TODO blocks in this file.

Requirements:
- Use a very small fully connected neural network (MLP), not CNN.
- Input is MNIST 28x28 grayscale images.
- Output should classify digits 0~9.
- Use CrossEntropyLoss.
- Use Adam optimizer.
- Keep the code simple and readable.
- Explain each block briefly.
- Do not add unnecessary advanced techniques."

You can modify this prompt based on what you want to try.
"""

# ============================================================
# BLOCK 0 - Import packages
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# BLOCK 1 - Basic configuration
# ============================================================
# TODO:
# You may ask an AI Agent what these values mean and whether
# these are reasonable starting values.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

print(f"Using device: {DEVICE}")


# ============================================================
# BLOCK 2 - Download / Load MNIST
# ============================================================
# This block is already completed for you.
#
# MNIST:
# - image size: 28 x 28
# - grayscale
# - labels: 0 ~ 9
#
# ToTensor() converts an image into a PyTorch tensor.
# The dataset will be downloaded automatically if it does not exist.

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples : {len(test_dataset)}")


# ============================================================
# BLOCK 3 - DataLoader / Batch
# ============================================================
# TODO:
# Create train_loader and test_loader.
#
# Think about:
# 1. What does batch_size mean?
# 2. Why should training data usually use shuffle=True?
# 3. Why can test data use shuffle=False?
#
# Expected variables:
#
# train_loader = ...
# test_loader = ...
#
# Hint:
# DataLoader(dataset, batch_size=..., shuffle=...)

# TODO: complete this block
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)


# ============================================================
# BLOCK 4 - Define a minimal neural network
# ============================================================
# TODO:
# Design a VERY SMALL fully connected neural network (MLP).
#
# Recommended structure:
#
# 28 x 28 image
#      |
#      v
# Flatten
#      |
#      v
# 784 values
#      |
#      v
# Linear layer
#      |
#      v
# ReLU
#      |
#      v
# Linear layer
#      |
#      v
# 10 outputs
#
# A simple example architecture is:
#
# 784 -> 128 -> 10
#
# Questions you should be able to answer:
# 1. Why is the input dimension 784?
# 2. Why is the output dimension 10?
# 3. What does the hidden layer do?
# 4. What is ReLU?
#
# TODO: complete this class

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Convert each 28 x 28 image into 784 input features.
        self.flatten = nn.Flatten()

        # Small MLP: 784 inputs -> 128 -> 64 hidden units -> 10 class scores.
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


# ============================================================
# BLOCK 5 - Create model
# ============================================================
# TODO:
# Create the model and move it to DEVICE.
#
# Expected variable:
#
# model = ...


model = SimpleMLP().to(DEVICE)


# ============================================================
# BLOCK 6 - Loss Function
# ============================================================
# TODO:
# Choose a loss function for 10-class classification.
#
# Recommended:
# CrossEntropyLoss
#
# You should be able to explain:
# "Loss tells us how wrong the current prediction is."
#
# Expected variable:
#
# criterion = ...


criterion = nn.CrossEntropyLoss()


# ============================================================
# BLOCK 7 - Optimizer / Learning Rate
# ============================================================
# TODO:
# Create an optimizer.
#
# Recommended:
# Adam
#
# Use LEARNING_RATE defined at the top.
#
# You should be able to explain:
# - What does the optimizer do?
# - What does learning rate control?
#
# Expected variable:
#
# optimizer = ...


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# BLOCK 8 - Training Loop
# ============================================================
# TODO:
# Complete the training loop.
#
# For each batch, the basic sequence should be:
#
# 1. Move images and labels to DEVICE
# 2. Clear old gradients
# 3. Forward
# 4. Calculate loss
# 5. Backward
# 6. Update model parameters
#
# In code, you should eventually have something similar to:
#
# optimizer.zero_grad()
# outputs = model(images)
# loss = criterion(outputs, labels)
# loss.backward()
# optimizer.step()
#
# BUT:
# Do not just copy these lines.
# Make sure you can explain what each one does.



def train_model(model, train_loader, criterion, optimizer, epochs):
    """Train the model and return the average loss for each epoch."""
    model.train()
    training_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in train_loader:
            # 1. Move images and labels to the same device as the model.
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 2. Clear gradients from the previous batch.
            optimizer.zero_grad()

            # 3. Forward pass: predict class scores.
            outputs = model(images)

            # 4. Compute classification loss.
            loss = criterion(outputs, labels)

            # 5. Backward pass: compute gradients.
            loss.backward()

            # 6. Update the model parameters.
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        training_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")

    return training_losses


def plot_training_loss(training_losses):
    """Save and display the training loss curve."""
    epochs = range(1, len(training_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, training_losses, marker="o", linewidth=2)
    ax.set_title("MNIST Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Training Loss")
    ax.set_xticks(list(epochs))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(__file__).with_name("training_loss.png")
    fig.savefig(output_path, dpi=150)
    print(f"Training loss plot saved to: {output_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# BLOCK 9 - Inference / Accuracy
# ============================================================
# This block is already completed for you.
#
# IMPORTANT:
# During inference:
# - We do NOT call loss.backward()
# - We do NOT call optimizer.step()
# - Model parameters are NOT updated
#
# We only:
#
# test image
#     |
#     v
# trained model
#     |
#     v
# prediction
#     |
#     v
# compare with correct label
#     |
#     v
# accuracy


@torch.no_grad()
def evaluate_accuracy(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        # outputs shape: [batch_size, 10]
        # argmax(dim=1) selects the class with the largest score.
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return accuracy


# ============================================================
# BLOCK 10 - Main Program
# ============================================================
# When all TODO blocks are completed, this section should:
#
# 1. Train the model
# 2. Run inference on the test dataset
# 3. Print final test accuracy


def main():

    # Simple checks to make unfinished blocks easier to identify.
    if train_loader is None or test_loader is None:
        raise RuntimeError(
            "BLOCK 3 is not finished: create train_loader and test_loader."
        )

    if model is None:
        raise RuntimeError(
            "BLOCK 5 is not finished: create the model."
        )

    if criterion is None:
        raise RuntimeError(
            "BLOCK 6 is not finished: define the loss function."
        )

    if optimizer is None:
        raise RuntimeError(
            "BLOCK 7 is not finished: define the optimizer."
        )

    # ----------------------------
    # Training
    # ----------------------------
    training_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    # ----------------------------
    # Inference / Accuracy
    # ----------------------------
    test_accuracy = evaluate_accuracy(
        model=model,
        test_loader=test_loader,
    )

    print("\n==============================")
    print("Final Result")
    print("==============================")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    plot_training_loss(training_losses)

    # This is NOT an accuracy competition.
    # A reasonable result with a simple MLP is enough.
    #
    # The important part is that you can explain:
    # Dataset -> Batch -> Model -> Loss -> Backward
    # -> Optimizer -> Epoch -> Inference -> Accuracy


if __name__ == "__main__":
    main()
