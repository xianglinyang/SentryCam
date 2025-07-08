import torch.nn as nn
import torch.nn.functional as F

# __all__ = ["mlp3"]

class MLP(nn.Module):
    def __init__(self, hidden_layer):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(hidden_layer,10)
        
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def mlp3():
    return MLP(512)