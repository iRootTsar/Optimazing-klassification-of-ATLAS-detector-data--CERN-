# ==== Imports
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


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
        x_flipped_horizontal = torch.flip(x, [3]) #flip horizontally
        x_flipped_vertical = torch.flip(x, [2]) #flip vertically
        x_rotated_180 = torch.rot90(x, 2, [2,3])#rotate 180
        
        x = F.relu(self.bn1(self.conv1(x)))
        x_flipped_horizontal = F.relu(self.bn1(self.conv1(x_flipped_horizontal)))
        x_flipped_vertical = F.relu(self.bn1(self.conv1(x_flipped_vertical)))
        x_rotated_180 = F.relu(self.bn1(self.conv1(x_rotated_180)))
        
        x = self.pool1(x)
        x_flipped_horizontal = self.pool1(x_flipped_horizontal)
        x_flipped_vertical = self.pool1(x_flipped_vertical)
        x_rotated_180 = self.pool1(x_rotated_180)
        
        x = (x + x_flipped_vertical + x_rotated_180 + x_flipped_horizontal) / 4
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


#Regular convModel

#Define the ConvModel class
class ConvModel(nn.Module):
    #Constructor that initializes the layers of the model
    def __init__(self, dropout):

        super(ConvModel, self).__init__()
        #opprette conv lag. Bildene har 3 lag 
        #Define convolutional layers for the input image
        #Input image has 3 challes, 3 layers
        #The first convolutional layer has 16 output channels and uses a kernel size of 3x3
        #ANd padding is set to 0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=0)
        #kanskje ha flere layers?

        #Unchange
        #Define fully connected layers
        #Input size to the first fully connected layer is 3*3*256
        #The first fully connected layer has 128 output units
        #The secon fully connected layer has 2 output units
        self.fc1 = nn.Linear(3*3*256, 128)
        self.fc2 = nn.Linear(128,2)

        #Define the dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)


    #Enten her eller over skal komme endringane for convolutional layer
    def forward(self, x:Tensor):
        
        #x = F.conv2d(x, self.conv1_filter, bias=self.conv1.bias, stride=1, padding=1)
        #Apply first conv layer
        x= self.conv1(x)
        #Apply relu activation function
        x = F.relu(x) 
        #Apply max pooling with a kernel size 2
        x = F.max_pool2d(x,2)
        #Apply the second conv layer
        x = self.conv2(x)
        #Apply relu activation function
        x = F.relu(x)
        #Apply max pooling with a kernel size of 2
        x = F.max_pool2d(x,2)
        #Apply the third conv layer
        x = self.conv3(x)
        #Apply relu activation function
        x = F.relu(x)
        #Appply max pooling with a kernel size 3
        x = F.max_pool2d(x,3)
        
        #Flatten the output of the conv layers
        x = torch.flatten(x, 1)
        
        #Apply the first fully connected layer
        x = self.fc1(x)
        #Apply relu activation function
        x = F.relu(x)
        x = self.dropout(x)
        #Apply teh second fully connected layer
        x = self.fc2(x)
        x = self.dropout(x)
        #Apply droput to layers??
        return x
#Convomodel that works when use data augmentation

class ConvModelAug(nn.Module):
    def __init__(self, dropout):
        super(ConvModelAug, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        
        # Determine the input shape for fc1 after the pooling layers
        def calc_output_shape(input_size, kernel_size, stride, padding):
            return (input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        input_size = 50
        input_size = calc_output_shape(input_size, 2, 2, 0)  # First max pool
        input_size = calc_output_shape(input_size, 2, 2, 0)  # Second max pool
        input_size = calc_output_shape(input_size, 2, 2, 0)  # Third max pool
        
        self.fc1 = nn.Linear(256 * input_size * input_size, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
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
        x_flipped_horizontal = torch.flip(x, [3])  # flip horizontally
        x_flipped_vertical = torch.flip(x, [2])  # flip vertically
        x_rotated_180 = torch.rot90(x, 2, [2, 3])  # rotate 180

        x = self.features(x)
        x_flipped_horizontal = self.features(x_flipped_horizontal)
        x_flipped_vertical = self.features(x_flipped_vertical)
        x_rotated_180 = self.features(x_rotated_180)

        x = (x + x_flipped_horizontal + x_flipped_vertical + x_rotated_180) / 4

        x = x.reshape(-1, 512 * 3 * 3)
        x = self.classifier(x)
        return x



#ResNet like model with first pooling layer adjustments

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
class ResSymmetricNet(nn.Module):
    def __init__(self, dropout):
        super(ResSymmetricNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)

        self.layer1 = BasicBlock(64, 128, stride=2)
        self.layer2 = BasicBlock(128, 256, stride=2)
        self.layer3 = BasicBlock(256, 512, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x_flipped_horizontal = torch.flip(x, [3])  # flip horizontally
        x_flipped_vertical = torch.flip(x, [2])  # flip vertically
        x_rotated_180 = torch.rot90(x, 2, [2, 3])  # rotate 180

        x = F.relu(self.bn1(self.conv1(x)))
        x_flipped_horizontal = F.relu(self.bn1(self.conv1(x_flipped_horizontal)))
        x_flipped_vertical = F.relu(self.bn1(self.conv1(x_flipped_vertical)))
        x_rotated_180 = F.relu(self.bn1(self.conv1(x_rotated_180)))

        x = self.pool1(x)
        x_flipped_horizontal = self.pool1(x_flipped_horizontal)
        x_flipped_vertical = self.pool1(x_flipped_vertical)
        x_rotated_180 = self.pool1(x_rotated_180)

        x = (x + x_flipped_horizontal + x_flipped_vertical + x_rotated_180) / 4
        x = self.dropout1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.reshape(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = self.fc2(x)
        return x