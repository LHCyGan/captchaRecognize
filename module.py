# -*- encoding:utf-8 -*-
# author: liuheng
import torch
from torch import nn
from resnet import ResNet18


class LstmCTCNet(nn.Module):
    def __init__(self, image_shape, label_map_length):
        super(LstmCTCNet, self).__init__()

        self.backbone = ResNet18()
        # 计算shape
        x = torch.zeros((1, 3) + image_shape)
        shape = self.backbone(x).shape
        bone_output_shape = shape[1] * shape[2]

        self.lstm = nn.LSTM(bone_output_shape, bone_output_shape, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(bone_output_shape * 2, label_map_length)

    def forward(self, x):
        x = self.backbone(x)

        x = x.permute(3, 0, 1, 2)
        w, b, c, h = x.shape
        x = x.view(w, b, c*h)

        x, _ = self.lstm(x)
        time_step, batch_size, hidden = x.shape
        x = x.view(time_step * batch_size, hidden)
        x = self.fc(x)

        return x.view(time_step, batch_size, -1)

if __name__ == '__main__':
    print(LstmCTCNet())
