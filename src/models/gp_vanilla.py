#%%
import random

import torch
from torch import nn
import matplotlib.pyplot as plt
import gp_functions as gp
import function_generator as fg
torch.pi = torch.acos(torch.zeros(1)).item() * 2

class GP_vanilla(nn.Module):
    def __init__(self):
        super().__init__()
        # Hyper-parameters
        l = torch.distributions.Uniform(1, 2).sample((1,))
        print(l)
        self.l = nn.Parameter(l, requires_grad=True)
        self.N = 200
        self.sigma2_f = 1
        self.sigma2_n = torch.tensor(0.0005)
        self.sigma_prior = 1
        self.total_no_param = 0

    def train_step(self, x, y, N, optimizer):
        optimizer.zero_grad(), self.zero_grad()
        loss = gp.NLML(x, y, N, self.l, self.sigma2_f, self.sigma2_n)
        loss.backward()
        optimizer.step()
        return loss

    def train_nn(self, x_space, y, EPOCHS, BATCH_SIZE, optimizer):
        for epoch in range(EPOCHS):
            idx = random.sample(range(self.N), BATCH_SIZE)
            x_train = x_space[idx, :]
            y_train = y[idx, :]
            loss = self.train_step(x_train, y_train, BATCH_SIZE, optimizer)
        return loss




m = GP_vanilla()
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)

y, x_space = fg.wall_pulse_func(1, 1, m.N, m.sigma2_n)
m.train_nn(x_space, y, EPOCHS=200, BATCH_SIZE=20, optimizer=opt)

yhat = gp.gp_uspace(m, x_space, y)
plt.scatter(x_space, yhat.data, 10, 'b')
plt.scatter(x_space, y, 10, 'g')
plt.show()