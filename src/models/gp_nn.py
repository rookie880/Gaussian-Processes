#%%
import random

import torch
from torch import nn
import matplotlib.pyplot as plt
import gp_functions as gp
import function_generator as fg
from torch.optim import Adam
from torch.utils.data import Dataset
torch.pi = torch.acos(torch.zeros(1)).item() * 2


#%%
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
        self.L3 = nn.Linear(self.L3_fi, self.L3_fo)

        # Hyper-parameters
        self.l = torch.sqrt(torch.tensor(1))
        self.N = 200
        self.sigma2_f = 1
        self.sigma2_n = torch.tensor(0.0005)
        self.sigma_prior = 1
        self.total_no_param = 0

        self.optm = Adam(self.parameters(), lr=0.001)


    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return x

    def train_step(self, x, y, N, optimizer):
        self.zero_grad()
        u = self.forward(x)
        loss = gp.NLML(u, y, N, self.l, self.sigma2_f, self.sigma2_n)
        loss.backward()
        optimizer.step()
        return loss

    def train_nn(self, EPOCHS, BATCH_SIZE):
        for epoch in range(EPOCHS):
            idx = random.sample(range(net.N), BATCH_SIZE)
            x_train = x_space[idx, :]
            y_train = y[idx, :]
            loss = self.train_step(x_train, y_train, BATCH_SIZE, self.optm)
        return loss




net = NN()
y, x_space = fg.wall_pulse_func(1, 1, net.N, net.sigma2_n)
net.train_nn(EPOCHS=2000, BATCH_SIZE=50)


# Plot observations
plt.plot(x_space, y)
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['y(x)'])
plt.grid()
plt.savefig('y.pdf')
plt.show()


uhat = net.forward(x_space)
yhat = gp.gp_uspace(net, uhat, y)
yhatx = gp.gp_uspace(net, x_space, y)

plt.scatter(x_space, yhat.data, 10, 'r')
plt.scatter(x_space, yhatx.data, 10, 'b')
plt.scatter(x_space, y, 10, 'g')
plt.grid()
plt.show()

plt.plot(x_space, uhat.data)
plt.grid()
plt.show()

