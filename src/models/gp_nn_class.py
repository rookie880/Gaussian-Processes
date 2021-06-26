#%%
import random
import numpy as np
import torch
from torch import nn
from src.models import gp_functions as gp
from torch.optim import Adam
torch.pi = torch.acos(torch.zeros(1)).item() * 2


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_fi = 1
        self.L1_fo = 32
        self.L2_fi = self.L1_fo
        self.L2_fo = 32
        self.L3_fi = self.L2_fo
        self.L3_fo = 1
        # self.l4_fi = self.L3_fo; self.l4_fo = 1

        self.structure = torch.tensor([[self.L1_fi, self.L1_fo], [self.L2_fi, self.L2_fo], [self.L3_fi, self.L3_fo]])

        self.L1 = nn.Linear(self.L1_fi, self.L1_fo)
        self.A1 = nn.Tanh()
        self.L2 = nn.Linear(self.L2_fi, self.L2_fo)
        self.A2 = nn.Tanh()
        self.L3 = nn.Linear(self.L3_fi, self.L3_fo, bias=False)

        # Hyper-parameters
        self.l = 0
        self.N = 0
        self.sigma2_f = 0
        self.sigma2_n = torch.tensor(0)
        self.total_no_param = 0

        # Optimizer
        self.optm = Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return x

    def train_step(self, x, y, N, optimizer):
        self.zero_grad()  # reset gradient
        u = self.forward(x)  # perform forward pass to enter u-space
        loss = gp.NLML(u, y, N, self.l, self.sigma2_f, self.sigma2_n)  # evaluate negative log marginal likelihood (NLML)
        loss.backward()  # Back propagation
        optimizer.step()  # steepest decent step.
        return loss

    def train_nn(self, x_space, y, EPOCHS, BATCH_SIZE):
        loss = np.nan
        for epoch in range(EPOCHS):
            idx = random.sample(range(self.N), BATCH_SIZE)
            x_train = x_space[idx, :]
            y_train = y[idx, :]
            loss = self.train_step(x_train, y_train, BATCH_SIZE, self.optm)  # perform steepest descent
        return loss

    def get_param(self):
        theta_dict = self.state_dict()
        out = []
        for param_tensor in theta_dict:
            temp = theta_dict[param_tensor]
            out.append(temp.view(-1))
        out = torch.cat(out)
        return out

    def update_param(self, theta_param):
        # theta_param is a vector consisting of all neural network parameter
        # Load theta_param into the neural network parameter
        theta_dict = self.state_dict()
        c = 0  # index to monitor how far we are in theta_param
        # Construct a parameter dictionary from theta_param
        for param_tensor in theta_dict:
            s = theta_dict[param_tensor].size()  # Get size of current weight matrix
            n = s.numel()  # number of elements
            param = torch.reshape(theta_param[c:c + n], s)  # reshape the part of theta_param
            c = c + n
            theta_dict[param_tensor] = param  # store in the parameter dictionary
        self.load_state_dict(theta_dict)  # update parameter dictionary

