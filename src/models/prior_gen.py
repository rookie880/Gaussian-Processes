# %% Libraries
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
        self.N = 200
        self.sigma2 = 1
        self.sigma2_n = torch.tensor(0.0001)
        self.sigma_prior = 1
        self.total_no_param = 0

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
        # temp = x - u # Bias toward M(x) = x = u
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


def se_kern(u_kern, l, sigma2, N):
    # out = torch.zeros([N,N])
    # l = 2*l*l
    # for i in range(N):
    #     for j in range(i,N):
    #         temp = sigma2*torch.exp(-(u_kern[i] - u_kern[j])**2/l)
    #         out[i, j] = temp
    #         out[j, i] = temp
    K_module.lengthscale = l
    K_module.outputscale = sigma2
    temp = K_module(u_kern)
    out = temp.evaluate()
    return out


def gp_prior(mod, u_prior):
    K_prior = se_kern(u_prior, mod.l, mod.sigma2, mod.N)
    B = K_prior + mod.sigma2_n * torch.eye(mod.N)
    f_prior = torch.distributions.MultivariateNormal(torch.zeros(mod.N), B)
    f_prior = f_prior.sample()
    omega = torch.normal(torch.zeros(mod.N), torch.ones(mod.N) * torch.sqrt(mod.sigma2_n))
    out = f_prior + omega
    out = torch.reshape(out, (mod.N, 1))
    return out


def gp_pred(mod, x_pred, y_pred):
    u_pred = mod.forward(x_pred)
    K_pred = se_kern(u_pred, mod.l, mod.sigma2, mod.N)
    # l = 2*mod.l*mod.l #K = mod.sigma2*torch.exp(-torch.cdist(u,u, p = 2)**2/l)

    B = K_pred + mod.sigma2_n * torch.eye(mod.N)
    L_pred = torch.cholesky(B)
    alpha_pred = torch.cholesky_solve(y_pred, L_pred)

    f_pred = K_pred @ alpha_pred

    ym = y_pred - f_pred
    LL_GP = torch.sum(ym ** 2) / mod.sigma2_n

    # Prior
    L_pred = torch.cholesky(K_pred + 1e-3 * torch.eye(mod.N))
    t1 = torch.cholesky_solve(f_pred, L_pred)
    LP_GP = t1.t() @ f_pred + torch.sum(torch.log(torch.diag(L_pred)))

    energy_gp = LL_GP + LP_GP
    return f_pred, energy_gp


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


def m_square_func(threshold, x, n):
    out = torch.zeros((n, 1))
    for i in range(n):
        if x[i] < threshold:
            out[i] = 1
        else:
            out[i] = 0
    return out


def gaussian_mixture_2(mean_well, well_distance, n):
    x = torch.zeros((n, 1))
    for i in range(n):
        if torch.rand(1) < 0.5:
            x[i] = torch.normal(mean_well, torch.tensor(well_distance))
        else:
            x[i] = torch.normal(-mean_well, torch.tensor(well_distance))
    return x


# %%
net = NN()
net.get_total_no_param()
net.prior()
theta_prior = net.get_param()

# Kernel Module #
K_module = gpytorch.kernels.RBFKernel()

# Generate Data and Plot #
x_space = gaussian_mixture_2(4, 1.0, net.N)
u = m_square_func(0, x_space, net.N)
y = gp_prior(net, u)

# Plot data #
plt.scatter(x_space, u.data, 20, 'r')
plt.show()
# Observations
plt.scatter(x_space, y, 20, 'b')
plt.legend(['M(x)=u', 'y=GP(u)'])
plt.show()

# %%
T = 10000
S = 0  # Number of samples
L = 5  # L = 3 ep = 0.0001 works (but it is to low for sure?.?)
ep0 = 0.01
M = 3  # Number of cycles
ep_space = ep0 * 0.5 * (torch.cos(
    torch.pi * torch.fmod(torch.linspace(0, T, T), torch.ceil(torch.tensor(T) / torch.tensor(M))) / torch.ceil(
        torch.tensor(T) / torch.tensor(M))) + 1)
plt.plot(ep_space)
plt.show()

fm = torch.zeros((net.N, 1))
um = torch.zeros((net.N, 1))
G = torch.zeros(T)

# Init
net.prior()
theta = net.get_param()
U_NN, ign = net.energy_nn(x_space, theta)
f, U_GP = gp_pred(net, x_space, y)
U = U_GP + U_NN
um_samples = []
grad_U = net.grad_calc(U)
x_test = torch.reshape(torch.linspace(-5, 5, 200), (200, 1))


# H-MCMC
for t in range(T):
    rp = torch.normal(torch.zeros(net.total_no_param), torch.ones(net.total_no_param))
    r = rp
    theta_p = theta
    grad_U_p = grad_U
    U_p = U

    # Leapfrog
    # ep = torch.rand(1)*0.01
    ep = ep_space[t]
    for i in range(L):
        rp = rp - ep * grad_U_p * 0.5
        theta_p = theta_p + ep * rp

        # Calculate gradient of U_p
        net.update_param(theta_p)
        U_NN_p, up = net.energy_nn(x_space, theta_p)  # NN pot. Energy
        f_p, U_GP_p = gp_pred(net, x_space, y)  # GP Pot. Energy
        U_p = U_GP_p + U_NN_p  # Proposed Pot. Energy
        grad_U_p = net.grad_calc(U_p)
        rp = rp - ep * grad_U_p * 0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / net.total_no_param  # Norm of Gradient
    K = torch.sum(r ** 2) / 2
    Kp = torch.sum(rp ** 2) / 2
    alpha = -U_p - Kp + U + K

    # print("Kp: ", Kp)
    # print("K: ", K)
    # print("Grad Norm: ", G[t] )
    # print("Ep: ", ep)

    if torch.log(torch.rand(1)) < alpha:
        theta = theta_p
        net.update_param(theta)
        U_NN = U_NN_p  # Neural Network potential energy update
        U_GP = U_GP_p  # GP potential Energy update
        U = U_p  # Potential Energy Update

        # print('Sample')
        if t > 500:
            fm = fm + f_p
            um = um + up
            S = S + 1
            u_test = net.forward(x_test)
            plt.plot(x_test, u_test.data, 'b', alpha=0.08)
            # torch.cat((um_samples, up))
    else:
        net.update_param(theta)
        # print('No Sample')
    print(t)
    print(S)
plt.show()
# %%
# Average over samples
f_mean = fm / S
u_mean = um / S

# Plot
# Latent space
plt.scatter(x_space, u.data, 20, 'r', 'o')
plt.legend(['M(x) = u'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()
plt.scatter(x_space, u_mean.data, 20, 'r', '*')
plt.legend(['Mhat(x) = uhat'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Targets
plt.scatter(x_space, y)
plt.scatter(x_space, f_mean.data)
plt.legend(['y', 'GP(u) = ybar', 'GP(uhat) = yhat'])

plt.xlabel('u')
plt.ylabel('y')
plt.show()

plt.plot(G)
plt.show()

'''
L = 5, ep1 = 0.0007, ep2 = 0.0005
OrderedDict([('L1.weight',
              tensor([[ 0.6725],
                      [-0.1466],
                      [-0.5422]])),
             ('L1.bias', tensor([ 0.4974, -1.1743, -0.6064])),
             ('L2.weight',
              tensor([[ 0.5291, -1.2355,  0.5454],
                      [-0.8836, -0.2579,  0.7238],
                      [-0.4228,  1.1979, -0.0870]])),
             ('L2.bias', tensor([-0.7052, -0.7126,  0.2960])),
             ('L3.weight', tensor([[-0.2347,  1.3749,  0.6065]])),
             ('L3.bias', tensor([0.8096]))])
'''
# %%
