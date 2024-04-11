import unittest
import torch
import torch.nn as nn
from collections import Counter
from torch.nn import functional as F

# Re-defining the classes here for testing purposes
class IndexDependentDense(nn.Module):
    def __init__(self, num_images, total_pixels, output_size, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.empty(num_images, total_pixels, output_size))
        self.b = nn.Parameter(torch.empty(num_images, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        return self.activation(y) if self.activation is not None else y

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = IndexDependentDense(num_images=4, total_pixels=25, output_size=256, activation=nn.ReLU())

    def forward(self, x):
        return self.dense(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = IndexDependentDense(num_images=4, total_pixels=256, output_size=1, activation=None)

    def forward(self, x):
        return torch.sigmoid(self.dense(x))


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(25, 256)

    def forward(self, x):
        return self.dense(x)

class MultiIonReadout(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def bceloss(self, X, y):
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true):
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        return (y_true == mod_y_pred).to(dtype=torch.float32).mean() * 100

    def accuracy(self, x, y):
        return self._accuracy(self(x), y)

# Test cases
class TestModelComponents(unittest.TestCase):
    def test_IndexDependentDense_output_shape(self):
        layer = IndexDependentDense(4, 25, 256)
        x = torch.randn(2, 4, 25)  # Simulate a batch of 2, 4 images each with 25 pixels
        output = layer(x)
        self.assertEqual(output.shape, (2, 4, 256))

    def test_Encoder_output_shape(self):
        encoder = Encoder()
        x = torch.randn(2, 4, 25)  # Simulate a batch of 2, 4 images each with 25 pixels
        output = encoder(x)
        self.assertEqual(output.shape, (2, 4, 256))

    def test_Classifier_output_shape(self):
        classifier = Classifier()
        x = torch.randn(2, 4, 512)  # Assume input feature size of 512 from encoder
        output = classifier(x)
        self.assertEqual(output.shape, (2, 4, 1))
        self.assertTrue(torch.all(output >= 0) & torch.all(output <= 1))

    def test_MultiIonReadout_integration(self):
        encoder = Encoder()
        classifier = Classifier()
        model = MultiIonReadout(encoder, classifier)
        x = torch.randn(2, 4, 5, 5)  # Simulate a batch of 2, each with 4 images of 5x5 pixels
        output = model(x)
        self.assertEqual(output.shape, (2, 4, 1))
        self.assertTrue(torch.all(output >= 0) & torch.all(output <= 1))

# Run the tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestModelComponents))
