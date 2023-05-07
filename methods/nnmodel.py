# ==== Imports
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
    

#Convo with first poolign layer adjustment with proper cylindricall symemtry using roll
# Custom pooling layer
def symmetry_pooling(x: Tensor, shift: int):
    x_lr = torch.flip(x, dims=[3])  # Left-right symmetry
    x_ud = torch.roll(x, shift, dims=[2])  # Top-bottom symmetry with displacement in y-axis
    x_lr_ud = torch.flip(x_ud, dims=[3])  # 180-degree rotation

    return (x + x_lr + x_ud + x_lr_ud) / 4

# Define the ConvModelFPL class
class ConvModelFPLRoll(nn.Module):
    def __init__(self, dropout):
        super(ConvModelFPLRoll, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1*1*1024, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = symmetry_pooling(x, shift=1)  # Apply the custom symmetry pooling function with shift
        x = F.avg_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.dropout(x)

        return x
    
#Convolutional layer with first pooling layer adjustment
# Custom pooling layer
def circular_shift(tensor: Tensor, shift: int, dim: int):
    return torch.cat((tensor[:, :, -shift:], tensor[:, :, :-shift]), dim=dim)

def symmetry_pooling(x: Tensor, shift: int):
    x_lr = torch.flip(x, dims=[3])  # Left-right symmetry
    x_ud = circular_shift(x, shift, dim=2)  # CIrcular shift for cyllindrical symmetry
    x_lr_ud = torch.flip(x_ud, dims=[3])  # 180-degree rotation

    return (x + x_lr + x_ud + x_lr_ud) / 4


# Define the ConvModelFPL class
class ConvModelFPLMod(nn.Module):
    def __init__(self, dropout):
        super(ConvModelFPLMod, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1*1*1024, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = symmetry_pooling(x, shift=1)  # Apply the custom symmetry pooling function with shift
        x = F.avg_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.dropout(x)

        return x

#Regular convModel
# Define the ConvModel class
class ConvModel(nn.Module):
    def __init__(self, dropout):
        super(ConvModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1*1*1024, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.dropout(x)

        return x



#CONVMODEL BASIC BITCH
#Define the ConvModel class
class ConvModelSimple(nn.Module):
    #Constructor that initializes the layers of the model
    def __init__(self, dropout):
        super(ConvModelSimple, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(3*3*256, 128)
        self.fc2 = nn.Linear(128,2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor):
        x = self.conv1(x)
        x = F.relu(x) 
        x = F.avg_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,3)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

#Symmetric net følgjer ikke noen konkret struktur til ResNet eller VGG net men har noen 
#karakteristikk til VGG med bruk av små konsolusjonslager med flere lag. 
#Og bruk av average poolign i staden for max pooling
class SymmetricNet(nn.Module):
    def __init__(self, dropout):
        super(SymmetricNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 128, 3) #first layer
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(128, 256, 4) #second layer
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv2d(256, 512, 3) #third layer
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout)
        
        self.conv4 = nn.Conv2d(512, 1024, 5, padding=2) #fourth layer
        self.bn4 = nn.BatchNorm2d(1024)
        self.pool4 = nn.AvgPool2d(2, 2)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(1024 * 2 * 2, 256)
        self.dropout5 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x_flipped_horizontal = torch.flip(x, [3]) # flip horizontally
        x_rotated_180 = torch.rot90(x, 2, [2, 3]) # rotate 180 degrees

        x = F.relu(self.bn1(self.conv1(x)))
        x_flipped_horizontal = F.relu(self.bn1(self.conv1(x_flipped_horizontal)))
        x_rotated_180 = F.relu(self.bn1(self.conv1(x_rotated_180)))

        x = self.pool1(x)
        x_flipped_horizontal = self.pool1(x_flipped_horizontal)
        x_rotated_180 = self.pool1(x_rotated_180)
    
        x_flipped_vertical = torch.roll(x, shifts=1, dims=2) # displace in the y-axis

        x = (x + x_flipped_horizontal + x_rotated_180 + x_flipped_vertical) / 4
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.reshape(-1, 1024 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)

        x = self.fc2(x)
        return x
    
#VGGNetBasic
class VGG_like(nn.Module):
    def __init__(self):
        super(VGG_like, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 25)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

#VGG architecture with first pooling layer adjustments

class VGGNet(nn.Module):
    def __init__(self, dropout):
        super(VGGNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)

        x = x.reshape(-1, 512 * 3 * 3)
        x = self.classifier(x)
        return x

#VGG updated
class VGGNet2(nn.Module):
    def __init__(self, dropout):
        super(VGGNet2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        shift_amount = x.size(2) // 2  # shift the image by half of the height
        x_shifted = torch.roll(x, shift_amount, 2)  # roll the image along the y-axis
        x_flipped_horizontal = torch.flip(x, [3])  # flip horizontally
        x_rotated_180 = torch.rot90(x, 2, [2, 3])  # rotate 180

        x = self.features(x)
        x_shifted = self.features(x_shifted)
        x_flipped_horizontal = self.features(x_flipped_horizontal)
        x_rotated_180 = self.features(x_rotated_180)

        x = (x + x_shifted + x_flipped_horizontal + x_rotated_180) / 4

        x = x.reshape(-1, 512 * 3 * 3)
        x = self.classifier(x)
        return x



#SymmetricNet Updated first pooling layer

class CircularAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CircularAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x_padded = F.pad(x, (0, 0, self.kernel_size // 2, self.kernel_size // 2), mode='circular')
        return F.avg_pool2d(x_padded, self.kernel_size, self.stride)

class SymmetricNet2(nn.Module):
    def __init__(self, dropout):
        super(SymmetricNet2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = CircularAvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(128, 256, 4) #second layer
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv2d(256, 512, 3) #third layer
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout)
        
        self.conv4 = nn.Conv2d(512, 1024, 5, padding=2) #fourth layer
        self.bn4 = nn.BatchNorm2d(1024)
        self.pool4 = nn.AvgPool2d(2, 2)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(1024 * 2 * 2, 256)
        self.dropout5 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x_flipped_horizontal = torch.flip(x, [3]) # flip horizontally
        x_rotated_180 = torch.rot90(x, 2, [2, 3]) # rotate 180
        
        x = F.relu(self.bn1(self.conv1(x)))
        x_flipped_horizontal = F.relu(self.bn1(self.conv1(x_flipped_horizontal)))
        x_rotated_180 = F.relu(self.bn1(self.conv1(x_rotated_180)))
        
        x = self.pool1(x)
        x_flipped_horizontal = self.pool1(x_flipped_horizontal)
        x_rotated_180 = self.pool1(x_rotated_180)
        
        x = (x + x_flipped_horizontal + x_rotated_180) / 3
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = x.reshape(-1, 1024 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        
        x = self.fc2(x)
        return x
    
#ResNet like model with first pooling layer adjustments
#It sucks

# Define the BasicBlock for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Updated ResNet-like SymmetricNet
class ResNet(nn.Module):
    def __init__(self, dropout):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)

        self.layer1 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer2 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer3 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        shift_amount = x.size(2) // 2  # shift the image by half of the height
        x_shifted = torch.roll(x, shift_amount, 2)  # roll the image along the y-axis
        
        x_flipped_horizontal = torch.flip(x, [3])  # flip horizontally
        x_rotated_180 = torch.rot90(x, 2, [2, 3])  # rotate 180

        x = F.relu(self.bn1(self.conv1(x)))
        x_shifted = F.relu(self.bn1(self.conv1(x_shifted)))
        x_flipped_horizontal = F.relu(self.bn1(self.conv1(x_flipped_horizontal)))
        x_rotated_180 = F.relu(self.bn1(self.conv1(x_rotated_180)))

        x = self.pool1(x)
        x_shifted = self.pool1(x_shifted)
        x_flipped_horizontal = self.pool1(x_flipped_horizontal)
        x_rotated_180 = self.pool1(x_rotated_180)

        x = (x + x_shifted + x_flipped_horizontal + x_rotated_180) / 4
        x = self.dropout1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.reshape(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.dropout3(x)

        x = self.fc3(x)
        return x
