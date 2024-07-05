import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.config import config
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from utilities.registry import MODELS
from utilities.visualize.visualize_results import plot_fitting_curve
from utilities.visualize.visualize_confusion_matrix import cm_plot
from torch.autograd import Function

from sklearn.metrics import confusion_matrix


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass
    def train_loop(self, dataloader, loss_fn, optimizer, neptune=None, field="train/loss"):
        self.train()
        loss_sum = 0
        size = len(dataloader.dataset)
        for batch, (X, _, y) in enumerate(dataloader):
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE, dtype=torch.float32)
            pred = self(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if neptune:
            neptune[f"{field}"].log(loss_sum / size)

    def test_loop(self, dataloader, loss_fn, neptune=None, field="test/loss"):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0
        self.eval()
        with torch.no_grad():
            for X, y, _ in dataloader:
                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= size
        print(f"Test Error: Avg loss: {test_loss:>8f}")
        if neptune:
            neptune[f"{field}"].log(test_loss)
        return test_loss

    def predict(self, dataloader):
        self.eval()
        pred = []
        target = []
        with torch.no_grad():
            for X, y, _ in dataloader:
                pred.append(self(X).cpu().detach().numpy())
                target.append(y.cpu().detach().numpy())
            target = np.concatenate(target).reshape(-1)
            pred = np.concatenate(pred).reshape(-1)
        return target, pred

    def evaluation(self, dataloader, neptune=None, field="test"):
        self.eval()
        pred = []
        target = []
        with torch.no_grad():
            for X, y, _ in dataloader:
                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)
                pred.append(self(X).cpu().detach().numpy())
                target.append(y.cpu().detach().numpy())
            target = np.concatenate(target).reshape(-1)
            pred = np.concatenate(pred).reshape(-1)
        fig = plot_fitting_curve(pred, target)
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        if neptune:
            neptune[f"{field}/rmse"].log(rmse)
            neptune[f"{field}/fitting_curve"].log(fig)

        return fig, rmse, target, pred


@MODELS.register()
class NN(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.layers[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        x = x.squeeze()
        x = self.sigmoid(x)
        return x


@MODELS.register()
class CNN(BaseModel):
    def __init__(self, input_dim=64 * 3, output_dim=1, hidden_dim=256, dropout=0.2, config=config):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        ws = config.WINDOW_SIZE
        params = config[f"cnn_params_ws_{ws}"]
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim,
                      kernel_size=params[0], stride=params[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout))
        self.cnn2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 4 * self.hidden_dim,
                      kernel_size=params[2], stride=params[3]),
            nn.ReLU(),
            nn.BatchNorm1d(4 * self.hidden_dim),
            nn.Dropout(self.dropout))
        self.cnn3 = nn.Sequential(
            nn.Conv1d(4 * self.hidden_dim, 6 * self.hidden_dim,
                      kernel_size=params[4], stride=params[5]),
            nn.ReLU(),
            nn.BatchNorm1d(6 * self.hidden_dim),
            nn.Dropout(self.dropout))
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.output = nn.Sigmoid()

        self.layers = [self.cnn, self.cnn2, self.cnn3, self.linear, self.linear2]

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.squeeze()
        x = self.linear(x)
        x = self.linear2(x)
        x = self.output(x)
        return x.squeeze()

    def freeze(self, layers=None):
        if layers is None:
            layers = [0, 1, 2]
        for layer in layers:
            for param in self.layers[layer].parameters():
                param.requires_grad = False


class CNNClassifier(CNN):
    def __init__(self, input_dim=64 * 3, output_dim=1, hidden_dim=256, dropout=0.2):
        super(CNNClassifier, self).__init__(input_dim, output_dim, hidden_dim, dropout)

    def predict(self, dataloader):
        self.eval()
        pred = []
        target = []
        with torch.no_grad():
            for X, _, y in dataloader:
                pred.append((self(X) > 0.5).to("cpu", dtype=torch.int))
                target.append(y)
            target = np.concatenate(target)
            pred = np.concatenate(pred)
        return target, pred

    def evaluation(self, dataloader, neptune=None, field="test"):
        target, pred = self.predict(dataloader)
        accuracy = np.mean(target == pred)
        fig = cm_plot(target, pred, title=f"Confusion Matrix. Average accuracy: {accuracy:.2f}")
        fig.show()
        return fig


class CNN2D(BaseModel):
    def __init__(self, config=config):
        super(CNN2D, self).__init__()
        self.config = config
        window_size = config.WINDOW_SIZE

        input_dim = window_size
        hidden_dim = 1024
        output_dim = 1

        params = config[f"cnn2d_params"]
        self.cnn1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=params[0], stride=params[1], padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=params[2], stride=params[3], padding=1),
            nn.ReLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=params[4], stride=params[5], padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def encoder(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze()
        x = self.decoder(x)
        return x.squeeze()

class DomainClassifier(nn.Module):
    """
    A wrapper for the run of encoder-rg-discriminator in order to run net in one
    back-propagation as described in paper.
    """

    def __init__(self, encoder):
        super(DomainClassifier, self).__init__()
        self.encoder = encoder
        self.discriminator = nn.Sequential(
            nn.Linear(config.FEATURE_NUM, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.lambd = 0

    def update_lambd(self, lambd):
        self.lambd = lambd
        GradReverse.lambd = self.lambd

    def forward(self, input):
        x = self.encoder(input)
        x = GradReverse.apply(x)
        x = torch.flatten(x, start_dim=1)
        x = self.discriminator(x)
        x = x.squeeze()
        return x

class GradReverse(Function):
    lambd = 0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * GradReverse.lambd

class Estimator(BaseModel):
    def __init__(self, input_dim=64 * 3, output_dim=1, hidden_dim=256, dropout=0.2):
        super(Estimator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        ws = config.WINDOW_SIZE
        params = config[f"cnn_params_ws_{ws}"]
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim,
                      kernel_size=params[0], stride=params[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout))
        self.cnn2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 4 * self.hidden_dim,
                      kernel_size=params[2], stride=params[3]),
            nn.ReLU(),
            nn.BatchNorm1d(4 * self.hidden_dim),
            nn.Dropout(self.dropout))
        self.cnn3 = nn.Sequential(
            nn.Conv1d(4 * self.hidden_dim, self.hidden_dim,
                      kernel_size=params[4], stride=params[5]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout))
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(config.FEATURE_NUM, self.hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1)
        x = self.decode(x)
        return x.squeeze()

    def encode(self, x):
        x = self.cnn(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x

    def decode(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x