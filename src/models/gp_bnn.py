# %% Libraries
import random

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import gpytorch
import gp_functions as gp
import function_generator as fg
torch.pi = torch.acos(torch.zeros(1)).item() * 2


#%% Class
class BNN(nn.Module):
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
        # self.L1 = nn.Linear(self.L1_fi, self.L1_fo)
        # self.A1 = nn.ReLU()
        # self.L2 = nn.Linear(self.L2_fi, self.L2_fo)
        # self.A2 = nn.ReLU()
        # self.L3 = nn.Linear(self.L3_fi, self.L3_fo)
        # self.A3 = nn.ReLU()
        # self.L4 = nn.Linear(self.l4_fi, self.l4_fo)

        # Hyper-parameters
        self.l = torch.sqrt(torch.tensor(0.5))
        self.N = 200
        self.sigma2_f = 1
        self.sigma2_n = torch.tensor(0.0005)
        self.sigma_prior = 1
        self.total_no_param = 0

        # Aux variables

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        # x = self.A3(x)
        # u = self.L4(x)
        return x

    def prior(self):
        theta_sample = torch.normal(torch.zeros(self.total_no_param),
                                    torch.ones(self.total_no_param) * self.sigma_prior)
        self.update_param(theta_sample)
        return

    def get_total_no_param(self):
        theta_dict = self.state_dict()
        t1 = []
        for param_tensor in theta_dict:
            t2 = theta_dict[param_tensor]
            t1.append(t2.view(-1))
        t1 = torch.cat(t1)
        self.total_no_param = len(t1)
        return

    def energy_nn(self, x, theta_param):
        u_latent = self.forward(x)
        # temp = x - u # Bias toward M(x) = x = u, will only work for n = m
        temp = u_latent  # Bias toward M(x) = 0

        ll_nn = torch.sum(temp ** 2) / self.sigma_prior
        lp_nn = torch.sum(theta_param ** 2) / self.sigma_prior
        return (ll_nn + lp_nn), u_latent

    def update_param(self, theta_param):
        theta_dict = self.state_dict()
        c = 0
        for param_tensor in theta_dict:
            s = theta_dict[param_tensor].size()
            n = s.numel()
            param = torch.reshape(theta_param[c:c + n], s)
            c = c + n
            theta_dict[param_tensor] = param
        self.load_state_dict(theta_dict)

    def grad_calc(self, energy):
        self.zero_grad()
        temp = grad(energy, self.parameters())
        grads = []
        for g in temp:
            grads.append(g.view(-1))
        grads = torch.cat(grads)
        return grads

    def get_param(self):
        theta_dict = self.state_dict()
        out = []
        for param_tensor in theta_dict:
            temp = theta_dict[param_tensor]
            out.append(temp.view(-1))
        out = torch.cat(out)
        return out


bnn = BNN()
bnn.get_total_no_param()
bnn.prior()


# Init Kernel Module
K_module = gpytorch.kernels.RBFKernel()

# %% Generate Data
# Generate Data and Plot
#x_space = torch.cat((torch.linspace(-5, -1, 50), torch.linspace(1, 5, 50)))
#x_space = torch.linspace(-5, 5, bnn.N)
#x_space = torch.reshape(x_space, (bnn.N, 1))
#y = square_func(0, x_space, 1, bnn.N, bnn.sigma2_n)
y, x_space = fg.wall_pulse_func(1, 1, bnn.N, bnn.sigma2_n)
#x_space = torch.linspace(-5, 5, bnn.N)
#x_space = torch.reshape(x_space, (bnn.N,1))

# Observations

# Plot observations
plt.plot(x_space, y)
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['y(x)'])
plt.grid()
plt.savefig('y.pdf')
plt.show()

# %% Sampling init
T = 10000
s = 0  # Number of samples
e = 0  # Number of exploration stages
L = 10  # Leapfrog steps # L = 10, ep0 = 0.005 appear to work ok
alt_flag = False  # if true then turn on alternative posterior. using the marginal likelihood p(y|u)
M = 3  # Number of cycles
beta = 0.2  # Proportion of exploration stage, take beta proportion of each cyclic to use exploration only

ep_space, t_burn, poly, cyclic = fg.ep_generate(T, M, ep0=0.002, ep_max=0.01, ep_min=0.000002,
                                                gamma=0.99, t_burn=500, ep_type="Cyclic")

# HMCMC
bnn.prior()
theta = bnn.get_param()
U_nn, ign = bnn.energy_nn(x_space, theta)
f, U_gp = gp.gp(bnn, x_space, y, alt_flag)
U = U_gp + U_nn
grad_U = bnn.grad_calc(U)

x_interpolate = torch.reshape(torch.linspace(0, 10, 200), (200, 1))
u_interpolate_cum = 0*x_interpolate
f_cum = torch.zeros((bnn.N, 1))
u_cum = torch.zeros((bnn.N, 1))
G = torch.zeros(T)

for t in range(T):
    rp = torch.normal(torch.zeros(bnn.total_no_param), torch.ones(bnn.total_no_param))
    r = rp
    theta_p = theta
    grad_U_p = grad_U
    U_p = U

    # Leapfrog
    ep = ep_space[t]
    for i in range(L):
        rp = rp - ep * grad_U_p * 0.5
        theta_p = theta_p + ep * rp

        # Calculate gradient of U_p
        bnn.update_param(theta_p)
        U_nn_p, up = bnn.energy_nn(x_space, theta_p)  # NN pot. Energy
        fp, U_gp_p = gp.gp(bnn, x_space, y, alt_flag)  # GP Pot. Energy
        U_p = U_gp_p + U_nn_p  # Proposed Pot. Energy
        grad_U_p = bnn.grad_calc(U_p)
        rp = rp - ep * grad_U_p * 0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / bnn.total_no_param  # Norm of Gradient

    if (torch.fmod(torch.tensor(t-1), t_burn)/t_burn < beta and cyclic) or (t < t_burn and poly):
        #  Do exploration
        e += 1
        theta = theta_p
        bnn.update_param(theta)
        U_nn = U_nn_p  # Neural Network potential energy update
        U_gp = U_gp_p  # GP potential Energy update
        U = U_p  # Potential Energy Update
    else:
        #  Do sampling
        K = torch.sum(r ** 2) / 2
        Kp = torch.sum(rp ** 2) / 2
        alpha = -U_p - Kp + U + K
        if torch.log(torch.rand(1)) < alpha:
            theta = theta_p
            bnn.update_param(theta)
            U_nn = U_nn_p  # Neural Network potential energy update
            U_gp = U_gp_p  # GP potential Energy update
            U = U_p  # Potential Energy Update

            f_cum = f_cum + fp
            u_cum = u_cum + up
            s = s + 1
            u_test = bnn.forward(x_interpolate)
            u_interpolate_cum = u_interpolate_cum + u_test
            plt.plot(x_interpolate, u_test.data, 'b', alpha=0.02)
        else:
            bnn.update_param(theta)
    print(t, ' : ', s, ' : ', e)
plt.show()
# %% Show results. Average over samples
yhat = f_cum / s
uhat = u_cum / s
uhat_interpolate = u_interpolate_cum / s


# Plot Latent space
plt.plot(x_space, uhat.data)
plt.legend(['Mhat(x) = uhat'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Plot Latent space predictions
plt.plot(x_interpolate, uhat_interpolate.data)
plt.legend(['Mhat(x_interpolate) = uhat_interpolate'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()


# Plot Targets and filtered values yhat
plt.scatter(x_space, y)
plt.scatter(x_space, yhat.data)
plt.legend(['y', 'GP(uhat) = yhat'])

plt.xlabel('u')
plt.ylabel('y')
plt.show()

plt.plot(G)
plt.show()


# %%
