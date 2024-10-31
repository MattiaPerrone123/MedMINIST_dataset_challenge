import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def configure_training_components(model, lr, milestones, gamma, task, optimizer_type="adam"):
    """
    Sets up the criterion, optimizer, and scheduler for training based on the task type and hyperparameters
    """
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type. Choose either 'adam' or 'sgd'")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return criterion, optimizer, scheduler

def train(model, train_loader, task, criterion, optimizer, device, writer, iteration):
    """
    Trains the model for one epoch and logs the training loss
    """
    total_loss = []
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
        loss.backward()
        optimizer.step()
    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss

def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):
    """
    Evaluates the model on a given dataset, computes performance metrics, and returns loss, AUC, accuracy, and F1 score
    """
    model.eval()
    total_loss = []
    y_score = torch.tensor([]).to(device)
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
                predicted = (outputs > 0.5).float()
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                _, predicted = torch.max(outputs, 1)
            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        test_loss = sum(total_loss) / len(total_loss) 
        f1 = f1_score(all_targets, all_preds, average='weighted') 
        return [test_loss, auc, acc, f1]
