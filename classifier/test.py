from torchviz import make_dot
import torch
from models import Classifier
from main import model

N = 4
N_i = 25

sample_input = torch.randn(1, N, N_i)
output = model(sample_input)
dot = make_dot(output, params=dict(list(model.named_parameters())))
dot.format = "png"
dot.render("model_architecture2")
