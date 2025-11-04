import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, padding=2)
        self.conv2 = nn.Conv2d(64, 66, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(66, 68, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(68, 70, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(70, 72, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(72 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, 72 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# ✅ load the same custom model
loaded_model = AlexNet()
loaded_model.load_state_dict(torch.load("./results/AlexNet.pt", map_location="cpu"))
loaded_model.eval()
