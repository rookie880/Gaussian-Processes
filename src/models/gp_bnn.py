# %% Libraries
import random

import torch
from torch import nn
from src.models import gp_nn
import matplotlib.pyplot as plt
from torch.autograd import grad
import gpytorch
from src.models import gp_functions as gp
from src.models import function_generator as fg
import torch.nn.functional as tnf
torch.pi = torch.acos(torch.zeros(1)).item() * 2

K_module = gpytorch.kernels.RBFKernel()
def gp_pred(xobs, xstar, yobs, lengthscale, sigma2_f, sigma2_n, N):
    K_module.lengthscale = lengthscale*lengthscale
    temp = K_module(xstar, xobs)
    K_star_obs = sigma2_f*temp.evaluate()
    temp = K_module(xstar, xstar)
    K_star_star = sigma2_f*temp.evaluate()
    temp = K_module(xobs, xobs)
    K_obs_obs = sigma2_f*temp.evaluate()

    B = K_obs_obs+sigma2_n*torch.eye(N)
    alpha_pred, B_LU = torch.solve(yobs, B)
    fbar = K_star_obs @ alpha_pred

    return fbar


def linear_squeeze(x, lower, upper, flag):
    if flag:
        mi = torch.min(x)
        mx = torch.max(x)
        a = -(upper-lower)/(mi-mx)
        b = (mi+mx)/(mi-mx)
        fx = a*x + b
    else:
        fx = x
    return fx

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

        self.structure = torch.tensor([[self.L1_fi, self.L1_fo], [self.L2_fi, self.L2_fo], [self.L3_fi, self.L3_fo]])

        self.L1 = nn.Linear(self.L1_fi, self.L1_fo)
        self.A1 = nn.Tanh()
        self.L2 = nn.Linear(self.L2_fi, self.L2_fo)
        self.A2 = nn.Tanh()
        self.L3 = nn.Linear(self.L3_fi, self.L3_fo)


        # Hyper-parameters
        self.l = torch.sqrt(torch.tensor(3.0))
        self.N = 200
        self.sigma2_f = 1
        self.sigma2_n = torch.tensor(0.0005)
        self.sigma2_prior = 0.1
        self.sigma2_likelihood = 0.5
        self.total_no_param = 0

        # Aux variables

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return x

    def prior(self):
        theta_sample = torch.normal(torch.zeros(self.total_no_param),
                                    torch.ones(self.total_no_param) * self.sigma2_prior)
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

        ll_nn = torch.sum(temp ** 2) / self.sigma2_likelihood
        lp_nn = torch.sum(theta_param ** 2) / self.sigma2_prior
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

# Generate Data
#y, x_space = fg.wall_pulse_func(1, 1, bnn.N, bnn.sigma2_n)
thold_x = 2
x_space = torch.cat((torch.linspace(-5, -thold_x, 100), torch.linspace(thold_x, 5, 100)))
x_space = torch.reshape(x_space, (bnn.N, 1))
y = fg.square_func(threshold=0, x=x_space, amplitude=1, n=bnn.N, sigma2_n=bnn.sigma2_n)



# Plot observations
plt.scatter(x_space, y, 20, 'g')
plt.ylabel(r'$y$')
plt.xlabel(r'$x$')
plt.legend([r'$y(x)$'])
plt.title('Observed Square Function with sparsity around zero. $\sigma_n^2=0.0005$')
plt.grid()
plt.savefig('./Figures/y.pdf')
plt.show()

 # %% Sampling with warm start
T = 5000  # T = 20000, L = 5, alt_flag = True, M = 10, Beta = 0.2, ep0 = 0.0005
s = 0  # Number of samples
e = 0  # Number of exploration stages
L = 5  # T = 5000, L = 5, alt_flag = True, M = 2, Beta = 0.2, ep0 = 0.0003/0.0008
alt_flag = True  # if true then turn on alternative posterior. using the marginal likelihood p(y|u)
M = 2  # Number of cycles
beta = 0.2  # Proportion of exploration stage, take beta proportion of each cyclic to use exploration only

ep_space, t_burn, poly, cyclic = fg.ep_generate(T, M, ep0=0.002, ep_max=0.0008, ep_min=0.000002,
                                                gamma=0.99, t_burn=500, ep_type="Cyclic")

# Array init
x_interpolate = torch.reshape(torch.linspace(-5, 5, 200), (200, 1))
u_interpolate_cum = 0*x_interpolate
f_cum = torch.zeros((bnn.N, 1))
u_cum = torch.zeros((bnn.N, 1))
G = torch.zeros(T)
theta_norm = torch.zeros(T+1)

# HMCMC
net = gp_nn.NN()
net.train_nn(x_space=x_space, y=y, EPOCHS=2000, BATCH_SIZE=50)
plt.plot(x_interpolate, net.forward(x_interpolate).data, 'b')
plt.xlabel(r'$x$')
plt.ylabel(r'$M(x)=u$')
plt.grid()
plt.title(r'Initialization. Warm start $u$-space using Neural Network')
plt.savefig('./Figures/init_warm_start.pdf')
plt.show()
theta = net.get_param()
theta_norm[0] = torch.sum(theta**2)
U_nn, ign = bnn.energy_nn(x_space, theta)
f, U_gp = gp.gp(bnn, x_space, y, alt_flag)
U = U_gp + U_nn
grad_U = bnn.grad_calc(U)
u_samples = torch.tensor([])
f_samples = torch.tensor([])

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
        #theta_p = theta_p + ep * rp
        theta_p = theta_p + ep * rp


        # Calculate gradient of U_p
        bnn.update_param(theta_p)
        U_nn_p, up = bnn.energy_nn(x_space, theta_p)  # NN pot. Energy
        fp, U_gp_p = gp.gp(bnn, x_space, y, alt_flag)  # GP Pot. Energy
        U_p = U_gp_p + U_nn_p  # Proposed Pot. Energy
        grad_U_p = bnn.grad_calc(U_p)
        rp = rp - ep * grad_U_p * 0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / bnn.total_no_param  # Norm of Gradient
    theta_norm[t+1] = torch.sum(theta_p ** 2)
    if (torch.fmod(torch.tensor(t-1), t_burn)/t_burn < beta and cyclic) or (t < t_burn and poly):
        #  Do exploration
        e += 1  # exploration count'
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
            u_interpolate = bnn.forward(x_interpolate)
            u_interpolate_cum = u_interpolate_cum + u_interpolate
            u_samples = torch.cat((u_samples, u_interpolate), dim=1)
            plt.plot(x_interpolate, u_interpolate.data, 'b', alpha=0.02)

            fbar = gp_pred(up, u_interpolate, y, bnn.l, bnn.sigma2_f, bnn.sigma2_n, bnn.N)
            f_samples = torch.cat((f_samples, fbar), dim=1)



        else:
            bnn.update_param(theta)
    print(t, ' : ', s, ' : ', e)

plt.xlabel(r'$x$')
plt.ylabel(r'$M(x)=u$')
plt.title(r'Each sample, $M_i$, of the probabilistic mapping $M$')
plt.grid()
plt.savefig('./Figures/M_samples.pdf')
plt.show()
u_samples = u_samples.t()
f_samples = f_samples.t()
# %% Show results. Average over samples
lin_squeeze_flag = 0
yhat = f_cum / s
uhat = u_cum / s
uhat_interpolate = u_interpolate_cum / s

u_samples_store = u_samples
u_samples_box = u_samples
f_samples_box = f_samples
if lin_squeeze_flag:
    u_samples_max = torch.max(u_samples_box, dim=1).values
    u_samples_max = torch.reshape(u_samples_max, (s, 1))
    u_samples_min = torch.min(u_samples_box, dim=1).values
    u_samples_min = torch.reshape(u_samples_min, (s, 1))

    a = -2/(u_samples_min-u_samples_max)
    b = (u_samples_min+u_samples_max)/(u_samples_min-u_samples_max)
    u_samples_box = torch.add(torch.multiply(u_samples_box, a), b)


u_mean = torch.mean(u_samples_box, dim=0)
u_upper = u_mean + 1.96*torch.std(u_samples_box, dim=0).data
u_lower = u_mean - 1.96*torch.std(u_samples_box, dim=0).data
f_mean = torch.mean(f_samples_box, dim=0)
f_upper = f_mean + 1.96*torch.std(f_samples_box, dim=0).data
f_lower = f_mean - 1.96*torch.std(f_samples_box, dim=0).data
delta_lower_upper = torch.sum(torch.abs(f_upper-f_lower))/(thold_x*2)

plt.plot(x_interpolate, u_mean.data, 'b', label=r'$E[M]$')
plt.plot(x_interpolate, u_upper.data, '--b', label=r'$E[M]+\sqrt{V[M]}$')
plt.plot(x_interpolate, u_lower.data, '--b', label=r'$E[M]-\sqrt{V[M]}$')
plt.title(r'Estimated $E[M]=\hat{u}$ and estimated $E[M]\pm1.96\sqrt{V[M]}$')
plt.legend()
plt.grid()
if lin_squeeze_flag:
    plt.savefig('./Figures/estimated_mean_cf_lin_squeeze.pdf')
plt.savefig('./Figures/estimated_mean_cf.pdf')
plt.show()

plt.scatter(x_space, y, 15, 'g', label='Observations')
plt.plot(x_interpolate, f_mean.data, 'r', label=r'$\hat{y}$')
plt.plot(x_interpolate, f_upper.data, '--r', label=r'$\hat{y}+\sqrt{\hat{y}}$')
plt.plot(x_interpolate, f_lower.data, '--r', label=r'$\hat{y}-\sqrt{\hat{y}}$')
plt.title(r'$\hat{y}$ and $\hat{y}\pm1.96\sqrt{V[\hat{y}]}$')
plt.legend()
plt.grid()
if lin_squeeze_flag:
    plt.savefig('./Figures/estimated_f_cf_lin_squeeze.pdf')
plt.savefig('./Figures/estimated_f_cf.pdf')
plt.show()



#%%
fbar = gp_pred(linear_squeeze(uhat, -1, 1, lin_squeeze_flag), u_mean, y, bnn.l, bnn.sigma2_f, bnn.sigma2_n, bnn.N)
fbar_upper = gp_pred(linear_squeeze(uhat, -1, 1, lin_squeeze_flag), u_upper, y, bnn.l, bnn.sigma2_f, bnn.sigma2_n, bnn.N)
fbar_lower = gp_pred(linear_squeeze(uhat, -1, 1, lin_squeeze_flag), u_lower, y, bnn.l, bnn.sigma2_f, bnn.sigma2_n, bnn.N)

plt.scatter(x_space, y, 15, 'g', label='Observations')
plt.plot(x_interpolate, fbar.data, 'r', label=r'GP-fit on $\hat{u}$')
plt.plot(x_interpolate, fbar_upper.data, '--r', label=r'GP-fit on $\hat{u}+\sqrt{V[M]}$')
plt.plot(x_interpolate, fbar_lower.data, '--r',  label=r'GP-fit on $\hat{u}-\sqrt{V[M]}$')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
if lin_squeeze_flag:
    plt.savefig('./Figures/observations_interpolate_lin_squeeze.pdf')
plt.savefig('./Figures/observations_interpolate.pdf')
plt.grid()
plt.show()

#%%
# Plot Latent space
plt.plot(x_space, uhat.data)
plt.grid()
plt.legend(['Mhat(x) = uhat'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Plot Latent space predictions
plt.plot(x_interpolate, uhat_interpolate.data)
plt.grid()
plt.legend(['Mhat(x_interpolate) = uhat_interpolate'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Plot Targets and filtered values yhat
plt.scatter(x_space, y, 15, 'g')
plt.scatter(x_space, yhat.data, 15, 'r')
plt.grid()
plt.legend(['Observations', 'Fitted Values'])
plt.xlabel('x')
plt.ylabel('y')
if lin_squeeze_flag:
    plt.savefig('./Figures/observations_fit_lin_squeeze.pdf')
plt.savefig('./Figures/observations_fit.pdf')
plt.show()
