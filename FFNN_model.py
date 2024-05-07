import torch
import torch.nn as nn
import torch.nn.functional as F

class IndexDependentDense(nn.Module):
    def __init__(self, N, N_i, N_o, activation=nn.ReLU()):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.activation = activation

        # Initializing the parameters of the network
        self.register_parameter(
            "W", nn.Parameter(torch.empty(self.N, self.N_i, self.N_o))
        )
        self.register_parameter("b", nn.Parameter(torch.empty(self.N, self.N_o)))
        self._reset_parameters()
        pass

    def _reset_parameters(self): # Initiating the network with values
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x): # Defining the forward pass through the network using torch
        print("Shape of input tensor:", x.shape)
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        if self.activation is not None:
            return self.activation(y)
        else:
            return y

    pass

# ---------------------------------------------------------------------------------------------

class Encoder(nn.Module): #Creating an encoder using the above IndexDependentDense architecture 
    def __init__(self, N, N_i, N_o):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o

        self.dense = IndexDependentDense(N, N_i, N_o, activation=nn.ReLU())
        pass

    def forward(self, x):
        y = self.dense(x)
        return y

    pass


# ---------------------------------------------------------------------------------------------


class Classifier(nn.Module): #Creating a classifier using the above IndexDependentDense architecture 
    def __init__(self, N, N_i, N_o):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.dense = IndexDependentDense(N, N_i, N_o, activation=None)
        pass

    def forward(self, x):
        y = self.dense(x)
        y = torch.sigmoid(y)  # Apply sigmoid activation here
        return y

    pass


# ---------------------------------------------------------------------------------------------


class SharedEncoder(nn.Module): #Creating a shared encoder using default linear architecture 
    def __init__(self, N, N_i, N_o):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o

        self.dense = nn.Linear(N_i, N_o)
        pass

    def forward(self, x):
        y = self.dense(x)
        return y

    pass


# ---------------------------------------------------------------------------------------------


class MultiIonReadout(nn.Module):
    def __init__(self, encoder, shared_encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.shared_encoder = shared_encoder
        self.classifier = classifier

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32) #Reshaping the data for the forwards pass
        y1 = self.encoder(y)
        y2 = self.shared_encoder(y)
        y_concat = torch.cat([y1, y2], dim=-1) #Combining the data for each y node for returning the classification
        y = self.classifier(y_concat)
        return y

    def bceloss(self, X, y): #Setting binary cross entropy as the loss function
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true): #Defining the accuracy prediction metric
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        accuracy = (y_true == mod_y_pred).to(dtype=torch.float32).mean()
        return accuracy * 100

    def accuracy(self, x, y):
        return self._accuracy(self(x), y)
