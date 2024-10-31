import os
from collections import OrderedDict
from copy import deepcopy

from tensorboardX import SummaryWriter
from tqdm import trange
import medmnist

from models.initializer import initialize_model

from utils.model_io import load_model_if_exists, save_best_model, log_and_save_results
from utils.device_setup import setup_device
from utils.data_loader import prepare_output_dir, load_datasets, create_dataloaders, get_data_transform
from utils.training_testing import configure_training_components, train, test
from utils.visualization import plot_losses, compute_and_save_confusion_matrix





def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, lr, gamma, size, download, model_flag, resize, as_rgb, model_path, run, confusion_matrix=False, plot_epochs=False, optimizer_type="adam"):
    """
    Sets up and trains a model, evaluates its performance, saves the best model, and optionally generates plots and confusion matrices
    """
    
    iteration = 0  
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    
    info = medmnist.INFO[data_flag]
    task, n_channels, n_classes = info['task'], 3 if as_rgb else info['n_channels'], len(info['label'])
    
    device = setup_device(gpu_ids)
    output_root = prepare_output_dir(output_root, data_flag)
    
    data_transform = get_data_transform(resize) 
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset, val_dataset, test_dataset = load_datasets(DataClass, data_transform, download, as_rgb, size)
    train_loader, train_loader_at_eval, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    model = initialize_model(model_flag, n_classes, n_channels, resize).to(device)
    best_model = load_model_if_exists(model, model_path, device)
    
    
    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)
    
    criterion, optimizer, scheduler = configure_training_components(model, lr, milestones, gamma, task, optimizer_type)

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))
    best_auc, best_epoch, best_model = 0, 0, deepcopy(model)
    
    train_losses = []
    val_losses = []

    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer, iteration)

        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)    

        train_losses.append(train_loss)
        val_loss = val_metrics[0] 
        val_losses.append(val_loss)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        scheduler.step()

        logs = ['loss', 'auc', 'acc', 'f1-score']

        train_log_dict = OrderedDict(zip([f"train_{log}" for log in logs], train_metrics))
        val_log_dict = OrderedDict(zip([f"val_{log}" for log in logs], val_metrics))
        test_log_dict = OrderedDict(zip([f"test_{log}" for log in logs], test_metrics))


        for key, value in test_log_dict.items():
            writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_auc, best_epoch = cur_auc, epoch
            best_model = deepcopy(model)
            print(f'Current best AUC: {best_auc}, at epoch: {best_epoch}')

    save_best_model(best_model, output_root)
    log_and_save_results(data_flag, train_metrics, val_metrics, test_metrics, output_root)
    writer.close()

    if plot_epochs:
        plot_losses(train_losses, val_losses, output_root)

    if confusion_matrix:
        cm_path = os.path.join(output_root, "confusion_matrixes")
        os.makedirs(cm_path, exist_ok=True)
        
        compute_and_save_confusion_matrix(best_model, train_loader_at_eval, device, n_classes, cm_path, dataset_name="train")
        compute_and_save_confusion_matrix(best_model, val_loader, device, n_classes, cm_path, dataset_name="val")
        compute_and_save_confusion_matrix(best_model, test_loader, device, n_classes, cm_path, dataset_name="test")

    return {
        'train_log': train_log_dict,
        'val_log': val_log_dict,
        'test_log': test_log_dict
            }  




