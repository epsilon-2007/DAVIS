import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, in_channel = 1, latent_size = 128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(9216, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(self.pool(x))
        x = F.relu(self.fc(x))
        return x
    
class Net(nn.Module):
    def __init__(self, args, in_channel = 1, latent_size = 128, num_classes = 10, normalize = False):
        super(Net, self).__init__()
        latent_size = args.feat_dim
        num_classes = args.num_classes
        self.normalize = args.normalize
        self.encoder = Encoder(in_channel=in_channel, latent_size=latent_size)
        self.head = nn.Linear(latent_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        if self.normalize:
            x = F.normalize(x, dim=1)
        x = self.head(x)
        return x