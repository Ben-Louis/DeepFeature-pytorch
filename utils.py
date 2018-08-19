import torch

default_mean = [0.485, 0.456, 0.406]
default_std = [0.229, 0.224, 0.225]

class Normalization(torch.nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()

        mean = torch.FloatTensor(mean).view(-1, 1, 1)
        std = torch.FloatTensor(std).view(-1, 1, 1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def recover(self, x):
        return (x * self.std + self.mean).clamp(0, 1)


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x, layers):
        feats = []
        end = max(layers)
        for i, module in enumerate(self._modules):
            x = self._modules[module](x)
            if i in layers:
                feats.append(x)
            if i == end:
                break
        return feats
        