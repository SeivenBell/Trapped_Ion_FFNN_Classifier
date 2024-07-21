import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexDependentDense(nn.Module):  # Bias calculation and activation for the network
    def __init__(self, N, N_i, N_o, activation):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.activation = activation

        # Initializing the parameters of the network
        self.register_parameter(
            "W", nn.Parameter(torch.empty(self.N, self.N_i, self.N_o))
        )
        self.register_parameter(
            "b", nn.Parameter(torch.empty(1, self.N, self.N_o))
        )  # I added batch dim 1
        self._reset_parameters()

    def _reset_parameters(self):  # Initiating the network with values
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):  # Defining the forward pass through the network
        # print("IndexDependentDense Shape of input 'X' tensor before reshape :", x.shape)
        x = x.float()
        x = x.reshape(-1, self.N, self.N_i)
        # print("IndexDependentDense Shape of input 'X' tensor after :", x.shape)
        y = (
            torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        )  # "..." remove and add just b
        # print("IndexDependentDense Shape of output 'Y' tensor in :", y.shape)
        if self.activation:
            y = self.activation(y)
        return y


# ---------------------------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, N, N_i, N_h):
        super().__init__()
        self.dense = IndexDependentDense(
            N, N_i, N_h, activation=nn.ReLU()
        )  # torch.relu?

    def forward(self, x):
        # print("YYEncoder Input tensor shape:", x.shape)  # Print input shape
        output = self.dense(x)
        # print("Encoder Output tensor shape:", output.shape)  # Print output shape
        return output


# ---------------------------------------------------------------------------------------------


class SharedEncoder(nn.Module):
    def __init__(self, N_c, N_h):
        super().__init__()
        self.dense = nn.Linear(N_c, N_h)
        self.dense2 = nn.Linear(N_c, N_h)

    def forward(self, x):
        return self.dense(F.relu(self.dense(x)))


# ---------------------------------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, N, N_h, N_o):
        super().__init__()
        self.dense = IndexDependentDense(N, N_h, N_o, activation=nn.Sigmoid())

    def forward(self, x):
        x = self.dense(x)
        # print("Classifier Output tensor shape:", x.shape)
        return x


# ---------------------------------------------------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        encoded = self.encoder(x)
        result = self.classifier(encoded)  # Classifying each group separately
        return result

    # def bceloss(self, X, y):
    #     # Forward pass to get output logits
    #     logits = self(X)
    #     return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    # def accuracy(self, X, y):
    #     logits = self(X)
    #     probs = torch.sigmoid(logits)
    #     # Convert probabilities to 0 or 1 based on a threshold of 0.5
    #     preds = (probs > 0.5).float()
    #     return (preds == y).float().mean()

    def bceloss(self, X, y):  # Setting binary cross entropy as the loss function
        # print("Shape of input tensor in MultiIonReadout:", X.shape)
        # print("Shape of output tensor in MultiIonReadout:", y.shape)
        X = X.float()
        y = y.float()
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true):  # Defining the accuracy prediction metric
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        accuracy = (y_true == mod_y_pred).to(dtype=torch.float32).mean()
        return accuracy * 100

    def accuracy(self, x, y):
        return self._accuracy(self(x), y)


class MultiIonReadout(nn.Module):
    def __init__(self, encoder, shared_encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.shared_encoder = shared_encoder
        self.classifier = classifier

    def forward(self, x):
        # print("MultiIonReadout Shape of input tensor 'X', just entry:", x.shape)
        y = x.reshape(*x.shape[:-2], -1).to(
            torch.float32
        )  # Reshaping the data for the forwards pass
        # print(
        #     "MultiIonReadout Shape of input tensor 'y' reshaped -> Encoder and Shared_Encoder:",
        #     y.shape,
        # )
        y1 = self.encoder(y)
        # print(
        #     "Shape of input tensor 'y1' in MultiIonReadout -> Shared_Encoder:", y1.shape
        # )

        y2 = y1.reshape(*y1.shape[:-2], -1)
        y3 = self.shared_encoder(y2)
        # print("Shape of input tensor 'y2' in MultiIonReadout -> Classifier:", y2.shape)
        y3 = y3.reshape(*y1.shape)

        y = y1 + y3

        # print(
        #     "Shape of input tensor 'y_concat' in MultiIonReadout -> Classifier:",
        #     y_concat.shape,
        #     "\n",
        # )
        y = self.classifier(y)

        return y

    def bceloss(self, X, y):  # Setting binary cross entropy as the loss function
        # print("Shape of input tensor in MultiIonReadout:", X.shape)
        # print("Shape of output tensor in MultiIonReadout:", y.shape)
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true):  # Defining the accuracy prediction metric
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        accuracy = (y_true == mod_y_pred).to(dtype=torch.float32).mean()
        return accuracy * 100

    def accuracy(self, x, y):
        return self._accuracy(self(x), y)
