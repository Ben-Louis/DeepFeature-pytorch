# DeepFeature-pytorch
Extract features using pre-trained deep CNNs



## Demo

`python extractor.py`



## Use Extractor in python

```python
from extractor import DeepFeature
"""
available base_model (temporarily):
'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
"""
extractor = DeepFeature(base_model='vgg19')

# make sure x is a 4-D tensor with range [0,1]
# stage n corresponds to the output of layer 'relu_n'
features = extractor(x, stage=[3,4,5])
# features will be a list that contains 3 tensors
```

