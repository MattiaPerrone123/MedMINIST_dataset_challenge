import torch
import os

def load_model_if_exists(model, model_path, device):
    """
    Loads a pre-trained model from the specified path if provided
    """
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
    return model

def save_best_model(model, output_root):
    """
    Saves the model's state dictionary as 'best_model.pth' in the specified output directory
    """
    state = {'net': model.state_dict()}
    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

def log_and_save_results(data_flag, train_metrics, val_metrics, test_metrics, output_root):
    """
    Logs the metrics of train, validation, and test datasets and saves them in a text file
    """
    log = (
        f"{data_flag}\n"
        f"train auc: {train_metrics[1]:.5f} acc: {train_metrics[2]:.5f} f1-score: {train_metrics[3]:.5f}\n"
        f"val auc: {val_metrics[1]:.5f} acc: {val_metrics[2]:.5f} f1-score: {val_metrics[3]:.5f}\n"
        f"test auc: {test_metrics[1]:.5f} acc: {test_metrics[2]:.5f} f1-score: {test_metrics[3]:.5f}\n"
    )
    print(log)
    with open(os.path.join(output_root, f'{data_flag}_log.txt'), 'a') as f:
        f.write(log)