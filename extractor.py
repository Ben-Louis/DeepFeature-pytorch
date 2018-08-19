import torch
import torch.nn as nn
import torchvision
from utils import Normalization, FeatureExtractor


class DeepFeature(nn.Module):
    def __init__(self, base_model='vgg19'):
        super(DeepFeature, self).__init__()

        # build model
        vgg19_model = getattr(torchvision.models, base_model)(pretrained=True)
        self.cnn_temp = vgg19_model.features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1
        self.stage2layer = {}

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                if relu_counter == 1:
                    self.stage2layer[block_counter] = i
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.MaxPool2d((2, 2), ceil_mode=True))  # ***


            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter) + "__" + str(i)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

        self.model.eval()

        # normalization
        self.norm = Normalization()


    def get_feat_with_layer(self, x, layers=None):
        x = self.norm(x).contiguous()
        if layers is None:
            layers = list(range(len(self.model)))
        return self.model(x, layers)

    def forward(self, x, stages=[1,2,3,4,5]):
        layers = [self.stage2layer[s] for s in stages]
        features = self.get_feat_with_layer(x, layers)
        return features


if __name__ == '__main__':
    from PIL import Image
    img = torchvision.transforms.ToTensor()(Image.open('image.jpg')).unsqueeze(0)

    extractor = DeepFeature('vgg19')
    feats = extractor(img)

    for f in feats:
        print(f.shape)