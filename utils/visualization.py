import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compute_and_save_confusion_matrix(model, data_loader, device, n_classes, save_path, dataset_name="test"):
    """
    Evaluates model predictions on a dataset, computes a confusion matrix, and saves it as a heatmap image
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)
    labels = list(range(n_classes))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Set')
    save_file = os.path.join(save_path, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(save_file)
    plt.close()

def plot_losses(train_losses, val_losses, output_path):
    """
    Plots and saves training and validation loss over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plot_file_path = os.path.join(output_path, "loss_plot.png")
    plt.savefig(plot_file_path)
    plt.close()
