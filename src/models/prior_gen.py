# %% Libraries
import random

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import gpytorch
torch.pi = torch.acos(torch.zeros(1)).item() * 2


# %% Functions
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_fi = 1
        self.L1_fo = 4
        self.L2_fi = self.L1_fo
        self.L2_fo = 4
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
        self.l = torch.sqrt(torch.tensor(0.1))
        self.N = 100
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


def se_kern(u_kern, lengthscale, sigma2_f):
    #temp = u_kern.flatten()
    #out = sigma2_f*torch.exp(-(temp[None, :] - temp[:, None])**2/l)
    K_module.lengthscale = lengthscale*lengthscale
    temp = K_module(u_kern)
    out = sigma2_f*temp.evaluate()
    return out


def gp_prior(mod, u_prior):
    K_prior = se_kern(u_prior, mod.l, mod.sigma2_f)
    B = K_prior + mod.sigma2_n * torch.eye(mod.N)
    f_prior = torch.distributions.MultivariateNormal(torch.zeros(mod.N), B)
    f_prior = f_prior.sample()
    eta = torch.normal(torch.zeros(mod.N), torch.ones(mod.N) * torch.sqrt(mod.sigma2_n))
    out = f_prior + eta
    out = torch.reshape(out, (mod.N, 1))
    return out


def gp(mod, x_pred, y_pred, alt_flag):
    u_pred = mod.forward(x_pred)
    K_pred = se_kern(u_pred, mod.l, mod.sigma2_f)
    B = K_pred + mod.sigma2_n * torch.eye(mod.N)

    # Not Cholesky. avoiding K^{-1}
    if not alt_flag:
        alpha_pred, B_LU = torch.solve(y_pred, B)
        f_pred = K_pred @ alpha_pred
        ym = y_pred - f_pred
        ll_gp = torch.sum(ym ** 2) / mod.sigma2_n
        _, K_pred_det = torch.linalg.slogdet(K_pred + 1e-4*torch.eye(mod.N))
        lp_gp = f_pred.t() @ alpha_pred + K_pred_det
        out = lp_gp + ll_gp
    else:
        alpha_pred, B_LU = torch.solve(y_pred, B)
        _, B_pred_det = torch.linalg.slogdet(B)
        out = y.t() @ alpha_pred + B_pred_det
        f_pred = K_pred @ alpha_pred

    return f_pred, out


def gp_uspace(mod, u_pred, y_pred):
    K_pred = se_kern(u_pred, mod.l, mod.sigma2_f)
    B = K_pred + mod.sigma2_n * torch.eye(mod.N)
    alpha_pred, B_LU = torch.solve(y_pred, B)
    yhat = K_pred @ alpha_pred
    return yhat


def m_spline_func(t1, t2, k, x, n):
    out = torch.zeros((n, 1), requires_grad=True)  # M(x) = u
    for i in range(n):
        if x[i] < t1:
            out[i] = k / (t1 - min(x)) * x[i] - k / (t1 - min(x)) * min(x)
        elif t1 <= x[i] < t2:
            out[i] = k
        else:
            out[i] = k / (max(x) - t2) * x[i] - k / (max(x) - t2) * t2 + k
    return out


def square_func(threshold, x, amplitude, n, sigma2_n):
    out = torch.zeros((n, 1))
    for i in range(n):
        if x[i] < threshold:
            out[i] = amplitude
        else:
            out[i] = -amplitude
    eta = torch.normal(torch.zeros(n, 1), torch.ones(n, 1)*torch.sqrt(sigma2_n))
    return out+eta


def gaussian_mixture_2(mean_well, well_distance, n):
    x = torch.zeros((n, 1))
    for i in range(n):
        if torch.rand(1) < 0.5:
            x[i] = torch.normal(torch.tensor(mean_well), torch.tensor(well_distance))
        else:
            x[i] = torch.normal(-torch.tensor(mean_well), torch.tensor(well_distance))
    return x


def ep_generate(T, M, ep0, ep_max, ep_min, gamma, t_burn, ep_type):
    ep_space = torch.zeros(T)
    poly = False
    cyclic = False
    if ep_type == "Cyclic":
        cyclic = True
        t_burn = torch.ceil(torch.tensor(T) / torch.tensor(M))
        ep_space = ep0 * 0.5 * (torch.cos(
            torch.pi * torch.fmod(torch.linspace(0, T, T), t_burn) / t_burn) + 1)
    elif ep_type == "Poly":
        poly = True
        for t in range(T):
            k = (ep_min/ep_max)**(1/gamma)
            b = k*T/(1-k)
            a = ep_max*b**gamma
            ep_space[t] = a*(b+t)**(-gamma)
    return ep_space, t_burn, poly, cyclic


# %% Generate Data
net = NN()
net.get_total_no_param()
net.prior()
theta_prior = net.get_param()

# Kernel Module #
K_module = gpytorch.kernels.RBFKernel()

# Generate Data and Plot
x_space = torch.cat((torch.linspace(-5, -1, 50), torch.linspace(1, 5, 50)))
#x_space = torch.linspace(-5, 5, net.N)
x_space = torch.reshape(x_space, (net.N, 1))
y = square_func(0, x_space, 1, net.N, net.sigma2_n)

# Observations
plt.scatter(x_space, y, 20, 'g')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['y(x)'])
plt.grid()
plt.savefig('y.pdf')
plt.show()

# %% Sampling
seed = 8
torch.manual_seed(seed)
random.seed(seed)

T = 5000
s = 0  # Number of samples
e = 0  # Number of exploration stages
L = 5  # Leapfrog steps # L = 10, ep0 = 0.005 appear to work ok
alt_flag = True  # if true then turn on alternative posterior. using the marginal likelihood p(y|u)
M = 5  # Number of cycles
beta = 0.1  # Proportion of exploration stage, take beta proportion of each cyclic to use exploration only

ep_space, t_burn, poly, cyclic = ep_generate(T, M, ep0=0.001, ep_max=0.1, ep_min=0.000002,
                                             gamma=0.99, t_burn=500, ep_type="Poly")

# Init
net.prior()
theta = net.get_param()
U_nn, ign = net.energy_nn(x_space, theta)
f, U_gp = gp(net, x_space, y, alt_flag)
U = U_gp + U_nn
grad_U = net.grad_calc(U)

x_interpolate = torch.reshape(torch.linspace(-5, 5, 200), (200, 1))
u_interpolate_cum = 0*x_interpolate
f_cum = torch.zeros((net.N, 1))
u_cum = torch.zeros((net.N, 1))
G = torch.zeros(T)

for t in range(T):
    rp = torch.normal(torch.zeros(net.total_no_param), torch.ones(net.total_no_param))
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
        net.update_param(theta_p)
        U_nn_p, up = net.energy_nn(x_space, theta_p)  # NN pot. Energy
        fp, U_gp_p = gp(net, x_space, y, alt_flag)  # GP Pot. Energy
        U_p = U_gp_p + U_nn_p  # Proposed Pot. Energy
        grad_U_p = net.grad_calc(U_p)
        rp = rp - ep * grad_U_p * 0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / net.total_no_param  # Norm of Gradient

    if (torch.fmod(torch.tensor(t-1), t_burn)/t_burn < beta and cyclic) or (t < t_burn and poly):
        #  Do exploration
        e += 1
        theta = theta_p
        net.update_param(theta)
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
            net.update_param(theta)
            U_nn = U_nn_p  # Neural Network potential energy update
            U_gp = U_gp_p  # GP potential Energy update
            U = U_p  # Potential Energy Update

            f_cum = f_cum + fp
            u_cum = u_cum + up
            s = s + 1
            u_test = net.forward(x_interpolate)
            u_interpolate_cum = u_interpolate_cum + u_test
            plt.plot(x_interpolate, u_test.data, 'b', alpha=0.02)
        else:
            net.update_param(theta)
    print(t, ' : ', s, ' : ', e)
plt.show()
# %% Show results

# Average over samples
yhat = f_cum / s
uhat = u_cum / s
uhat_interpolate = u_interpolate_cum / s


# Plot Latent space
plt.scatter(x_space, uhat.data, 20, 'r', '*')
plt.legend(['Mhat(x) = uhat'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Plot Latent space predictions
plt.scatter(x_interpolate, uhat_interpolate.data, 20, 'r', '*')
plt.legend(['Mhat(x) = uhat'])
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
