#%% Libraries
from os import RTLD_NODELETE
import numpy as np
from numpy.lib.function_base import rot90
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.overrides import get_overridable_functions
from torch.autograd import grad
from torch.quantization.quantize import propagate_qconfig_

#%% Functions
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_fi = 1; self.L1_fo = 4
        self.L2_fi = self.L1_fo; self.L2_fo = 4
        self.L3_fi = self.L2_fo; self.L3_fo = 1

        self.structure = torch.tensor([[self.L1_fi, self.L1_fo], [self.L2_fi, self.L2_fo], [self.L3_fi, self.L3_fo]])

        self.L1 = nn.Linear(self.L1_fi, self.L1_fo)
        self.A1 = nn.Tanh()
        self.L2 = nn.Linear(self.L2_fi, self.L2_fo)
        self.A2 = nn.Tanh()
        self.L3 = nn.Linear(self.L3_fi, self.L3_fo)

        # Hyperparameters
        self.l = torch.sqrt(torch.tensor(0.1)); self.N = 50; self.sigma2 = 1; self.sigma2_n = 0.001; self.sigma_prior = 1; self.total_no_param = 0

    def prior(self):
        theta_sample = torch.normal(torch.zeros(self.total_no_param), torch.ones(self.total_no_param)*self.sigma_prior)
        self.update_param(theta_sample)
        return 

    def get_total_no_param(self):
        theta_dict = self.state_dict()
        theta = []
        for param_tensor in theta_dict:
            temp = theta_dict[param_tensor]
            theta.append(temp.view(-1))
        theta = torch.cat(theta)
        self.total_no_param = len(theta)
        return 


    def U_NN(self, x, theta):
        u = self.forward(x)
        ll_NN = torch.tensor(0.0, requires_grad=True)
        #pvect = x - u # Bias toward M(x) = x = u
        pvect = u # Bias toward M(x) = 0
 
        ll_NN = torch.sum(pvect**2)/self.sigma_prior
        lp_NN = torch.sum(theta**2)/self.sigma_prior
        res = ll_NN + lp_NN
        return res, u

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        u = self.L3(x)
        return u

    def update_param(self, theta):
        theta_dict = self.state_dict()
        c = 0
        for param_tensor in theta_dict:
            s = theta_dict[param_tensor].size()
            n = s.numel()
            param = torch.reshape(theta[c:c+n], s)
            c = c + n
            theta_dict[param_tensor] = param
        self.load_state_dict(theta_dict)

    def grad_calc(self, U):
        self.zero_grad()
        temp = grad(U, self.parameters())
        grads = []
        for g in temp:
            grads.append(g.view(-1))
        grads = torch.cat(grads)
        grads = torch.reshape(grads, (self.total_no_param,1))
        return grads

    def get_param(self):
        theta_dict = self.state_dict()
        theta = []
        for param_tensor in theta_dict:
            temp = theta_dict[param_tensor]
            theta.append(temp.view(-1))
        theta = torch.cat(theta)
        theta = torch.reshape(theta, (self.total_no_param,1))
        return theta

def SE_kern(x, l, sigma2, N):
    K = torch.zeros([N,N])
    l = 2*l*l
    for i in range(N):
        for j in range(N):
            temp = (x[i] - x[j])**2
            K[i][j] = sigma2*torch.exp(-temp/l)
    return K

def GP_prior(u, l, sigma2, sigma2_n ,N):
    K = SE_kern(u, l, sigma2, N)
    B = K + sigma2_n*torch.eye(N)
    f = torch.distributions.MultivariateNormal(torch.zeros(N), B)
    f = f.sample()
    omega = torch.normal(torch.zeros(N), torch.ones(N)*sigma2_n)
    y = f + omega
    y = torch.reshape(y, (N, 1))
    return y

def GP_pred(mod, x, y):
    u = mod.forward(x)
    K = SE_kern(u, mod.l, mod.sigma2, mod.N)

    B = K + mod.sigma2_n*torch.eye(mod.N)
    L = torch.cholesky(B + 1e-5*torch.eye(mod.N))
    alpha, LU   = torch.solve(y,L)
    alpha, LU = torch.solve(alpha, torch.transpose(L, 0, 1))
    f = torch.matmul(K,alpha)

    ym = y - f
    LL_GP = 0.5*torch.sum(ym**2)/mod.sigma2_n

    # Prior
    L  = torch.cholesky(K + 1e-5*torch.eye(mod.N))
    t1, LU = torch.solve(f, L)
    t1, LU = torch.solve(t1, torch.transpose(L, 0, 1))
    t1 = torch.transpose(t1, 1, 0)
    LP_GP = 0.5*t1.matmul(f) + torch.sum(torch.log(torch.diag(L)))
    U_GP = LL_GP + LP_GP
    return f, U_GP

net = NN()
net.get_total_no_param()
net.prior()
theta_true = net.get_param()

# Generate Data and Plot
x_space = torch.linspace(-5, 5, net.N)
x_space = torch.reshape(x_space, (net.N,1))
u = net.forward(x_space)
y = GP_prior(u, net.l, net.sigma2, net.sigma2_n, net.N)
#% Plot data
# Latent space
plt.plot(x_space.data, u.data, c = 'red')
plt.show()
# Observations
plt.plot(x_space, y, c = 'blue')
plt.legend(['M(x)=u', 'y=GP(u)'])
plt.show()
#%% Sampling
net.prior()

T = 500
S = 0 # Number of samples
L = 5 # L = 3 ep = 0.0001 works (but it is to low for sure?.?)
G = torch.zeros(T)
ep1 = 0.0007
ep = ep1
fm = torch.zeros((net.N,1))
um = torch.zeros((net.N,1))

# Init
theta = net.get_param()
U_NN, ign = net.U_NN(x_space, theta)
f, U_GP = GP_pred(net, x_space, y)
U = U_GP + U_NN

grad_U = net.grad_calc(U)

for t in range(T):
    rp = torch.normal(torch.zeros(net.total_no_param), torch.ones(net.total_no_param))
    rp = torch.reshape(rp, (net.total_no_param,1))
    r = rp 
    theta_p = theta
    grad_U_p = grad_U
    U_p = U

    # Leapfrog
    for i in range(L):
        rp = rp - ep*grad_U_p*0.5
        theta_p = theta_p + ep*rp

        # Calculate gradient of U_p
        net.update_param(theta_p)
        U_NN_p, up = net.U_NN(x_space, theta_p) # NN pot. Energy
        f_p, U_GP_p = GP_pred(net, x_space, y) # GP Pot. Energy
        U_p = U_GP_p + U_NN_p # Proposed Pot. Energy
        grad_U_p = net.grad_calc(U_p)
        rp = rp - ep*grad_U_p*0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p**2))/net.total_no_param # Normalizes the gradient with the number of parameters to be sampled
    K = torch.sum(r**2)/2
    Kp = torch.sum(rp**2)/2
    alpha = -U_p - Kp + U + K
    #print("U_GP_p: ", U_GP_p)
    #print("U_NN_p: ", U_NN_p)
    print("Kp: ", Kp)

    #print("U_GP: ", U_GP)
    #print("U_NN: ", U_NN)
    print("K: ", K)
    print("Grad Norm: ", G[t] )

    if t > 40:
        ep = ep
        #ep = 0.0005

    
    if torch.log(torch.rand(1)) < alpha:
        theta = theta_p
        net.update_param(theta)
        U_NN = U_NN_p ## Neural Network potential energy update
        U_GP = U_GP_p  ## GP potential Energy update
        U = U_p ## Potential Energy Update

        print('Sample')
        if t > 40:
            fm = fm + f_p
            um = um + up
            S = S + 1
    else:
        net.update_param(theta)
        print('No Sample')
    print(t)
    print(S)

#%% 
# Average over samples
f_mean = fm/S
u_mean = um/S
net.update_param(theta_true)
ftrue, ign = GP_pred(net, x_space, y) # GP Energy

#%% Plot
# Latent space
plt.plot(x_space.data, (u.data) , c = 'red')
plt.legend(['M(x) = u'])
plt.xlabel('x')
plt.ylabel('u')
plt.show()
plt.plot(x_space.data, u_mean.data, 'r--')
plt.legend(['Mhat(x) = uhat'])
plt.title(('ep1 = ',str(ep1), ' ep2 = ',str(ep), 'L = ', str(L)))
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Targets
plt.plot(x_space.data, y , c = 'blue')
plt.plot(x_space.data, ftrue.data, 'm--')
plt.plot(x_space.data, f_mean.data, 'g--')
plt.legend(['y', 'GP(u) = ybar', 'GP(uhat) = yhat'])

plt.title(('ep1 = ',str(ep1), ' ep2 = ',str(ep), 'L = ', str(L)))
plt.xlabel('u')
plt.ylabel('y')
plt.show()

plt.plot(G)
plt.title(('||Grad_U_p||_2.', 'ep1 = ',str(ep1), ' ep2 = ',str(ep), 'L = ', str(L)))
#plt.ylim((min(G), torch.mean(G) + torch.std(G)))
plt.ylim((0, 350))
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
