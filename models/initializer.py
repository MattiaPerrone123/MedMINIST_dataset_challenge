from torchvision.models import resnet18, resnet50
from models.models import ResNet18, ResNet50

def initialize_model(model_flag, n_classes, n_channels, resize):
    """
    Initializes a ResNet18 or ResNet50 model, either custom or torchvision, based on the specified settings
    """
    if model_flag == 'resnet18':
        return resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        return resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    raise NotImplementedError("Model not supported")
