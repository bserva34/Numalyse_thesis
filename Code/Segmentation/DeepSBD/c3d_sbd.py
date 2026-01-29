import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D_SBD(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 96, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1,2,2), (1,2,2))

        self.conv2 = nn.Conv3d(96, 256, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((2,2,2), (2,2,2))

        self.conv3 = nn.Conv3d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(384, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d((2,2,2), (2,2,2))

        self.flatten_dim = self._get_flatten_size()

        self.fc6 = nn.Linear(self.flatten_dim, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, num_classes)

        self.drop = nn.Dropout(0.5)


    def forward(self, x, return_features=False):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = self.drop(x)

        x = F.relu(self.fc7(x))
        x = self.drop(x)

        if return_features:
            return x  # <-- 2048-D features (FC7)

        logits = self.fc8(x)
        return logits



    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        return x


    # def _get_flatten_size(self):
    #     with torch.no_grad():
    #         x = torch.zeros(1, 3, 16, 112, 112)
    #         x = self._forward_conv(x)
    #         return x.view(1, -1).shape[1]

    #modif seg 8
    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 8, 112, 112)
            x = self._forward_conv(x)
            return x.view(1, -1).shape[1]




#test
if __name__ == "__main__":
    model = C3D_SBD()
    x = torch.randn(2, 3, 16, 112, 112)
    y = model(x)
    print(y.shape)  # doit être (2, 3)

    print(torch.cuda.get_device_name())

