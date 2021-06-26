# %% Libraries
import numpy as np
import torch
from src.models import gp_nn_class
import matplotlib.pyplot as plt
from src.models import gp_bnn_class
import gpytorch
from src.models import gp_functions as gp
from src.models import function_generator as fg
torch.pi = torch.acos(torch.zeros(1)).item() * 2
K_module = gpytorch.kernels.RBFKernel()



# %% Generate Data
bnn = gp_bnn_class.BNN()
bnn.sigma2_likelihood = 1
bnn.l = torch.sqrt(torch.tensor(1.0))
bnn.N = 200
bnn.sigma2_f = 1
bnn.sigma2_n = torch.tensor(0.0005)
bnn.sigma2_prior = 1
bnn.get_total_no_param()
bnn.prior()

# Init Kernel Module
K_module = gpytorch.kernels.RBFKernel()

# Generate Data
#y, x_space = fg.wall_pulse_func(1, 1, bnn.N, bnn.sigma2_n)
thold_x = -1
#x_space = torch.cat((torch.linspace(-5, -thold_x, 100), torch.linspace(thold_x, 5, 100)))
#x_space = torch.reshape(x_space, (bnn.N, 1))
#y = fg.square_func(threshold=0, x=x_space, amplitude=1, n=bnn.N, sigma2_n=bnn.sigma2_n)

x_space = torch.cat((torch.linspace(-5, -thold_x, 100), torch.linspace(thold_x, 5, 100)))
x_space = torch.reshape(x_space, (bnn.N, 1))
y = torch.zeros((bnn.N, 1))
eta = fg.square_func(threshold=10, x=x_space, amplitude=1, n=bnn.N, sigma2_n=bnn.sigma2_n)
y[0:100, 0] = -1
y[100:bnn.N, 0] = 1
y = y + eta

# %% Sampling with warm start
T = 20000  # T = 20000, L = 5, alt_flag = True, M = 10, Beta = 0.2, ep0 = 0.0005
s = 0  # Number of samples
e = 0  # Number of exploration stages
L = 5  # T = 5000, L = 5, alt_flag = True, M = 2, Beta = 0.2, ep0 = 0.0003/0.0008
alt_flag = True  # if true then turn on posterior that use the marginal likelihood p(y|u)
M = int(T/50)  # Number of cycles
beta = 0.2  # Proportion of exploration stage, take beta proportion of each cyclic to use exploration only

ep_space, t_burn, poly, cyclic = fg.ep_generate(T, M, ep0=0.0008, ep_max=0.0008, ep_min=0.000002,
                                                gamma=0.99, t_burn=500, ep_type="Cyclic")

## HMCMC ##

# Warm start using neural network minimizing negative log marginal likelihood (NLML)
net = gp_nn_class.NN()
net.l = bnn.l; net.N = bnn.N; net.sigma2_f = bnn.sigma2_f; net.sigma2_n = bnn.sigma2_n  # Same hyper-parameters as the BNN
net.train_nn(x_space=x_space, y=y, EPOCHS=2000, BATCH_SIZE=50)
theta = net.get_param()  # Get the warm start parameters

# init energies and gradients
U_nn, _ = bnn.energy_nn(x_space, theta)
_, U_gp = gp.gp(bnn, x_space, y, alt_flag)
U = U_gp + U_nn
grad_U = bnn.grad_calc(U)

# Init tensors to store samples
u_samples = torch.zeros((T, bnn.N))  # u-space samples
f_samples = torch.zeros((T, bnn.N))  # Prediction from each u sample
G = torch.zeros(T)  # \grad E_U(theta) Gradient L2-norm for each t.
U_nn_p = np.nan; U_gp_p = np.nan; up = torch.zeros(bnn.N, 1)

# interpolating/prediction input points
x_interpolate = torch.reshape(torch.linspace(-5, 5, 200), (200, 1))


for t in range(T):
    rp = torch.normal(torch.zeros(bnn.total_no_param), torch.ones(bnn.total_no_param))
    r = rp
    theta_p = theta
    grad_U_p = grad_U
    U_p = U

    # Leapfrog
    ep = ep_space[t]  # Take current stepsize
    for i in range(L):

        rp = rp - ep * grad_U_p * 0.5  # Half step momentum
        theta_p = theta_p + ep * rp  # full step position/parameter

        # Calculate gradient of U_p (\gradE_U(theta'))
        bnn.update_param(theta_p)
        U_nn_p, up = bnn.energy_nn(x_space, theta_p)  # NN pot. Energy
        fp, U_gp_p = gp.gp(bnn, x_space, y, alt_flag)  # GP Pot. Energy
        U_p = U_gp_p + U_nn_p  # Proposed Pot. Energy
        grad_U_p = bnn.grad_calc(U_p)
        rp = rp - ep * grad_U_p * 0.5

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / bnn.total_no_param  # Norm of Gradient
    if (torch.fmod(torch.tensor(t-1), t_burn)/t_burn < beta and cyclic) or (t < t_burn and poly):
        #  Do exploration
        e += 1  # exploration steps counter
        theta = theta_p; bnn.update_param(theta)  # update parameters
        U_nn = U_nn_p  # Neural Network potential energy update
        U_gp = U_gp_p  # GP potential Energy update
        U = U_p  # Potential Energy Update
    else:
        #  Do sampling
        K = torch.sum(r ** 2) / 2
        Kp = torch.sum(rp ** 2) / 2
        alpha = -U_p - Kp + U + K  # Acceptance probability in log-space
        if torch.log(torch.rand(1)) < alpha:
            theta = theta_p; bnn.update_param(theta)  # Accept proposal theta_p
            U_nn = U_nn_p  # Neural Network potential energy update
            U_gp = U_gp_p  # GP potential Energy update
            U = U_p  # Potential Energy Update
            u_interpolate = bnn.forward(x_interpolate)
            u_samples[s, :] = u_interpolate.flatten()
            plt.plot(x_interpolate, u_interpolate.data, 'b', alpha=0.02)

            #fbar = gp.gp_pred(up, u_interpolate, y, bnn.l, bnn.sigma2_f, bnn.sigma2_n, bnn.N)
            f_hat = bnn.gp_pred(up, u_interpolate, y)
            f_samples[s, :] = f_hat.flatten()

            s = s + 1  # Number of samples accepted counter


        else:
            # No sample. Keep current position/parameter
            bnn.update_param(theta)
    print(t, ' : ', s, ' : ', e)  # Print progress

plt.xlabel(r'$x$')
plt.ylabel(r'$M(x)=u$')
plt.title(r'Each sample, $M_i$, of the probabilistic mapping $M$')
plt.grid()
plt.savefig('./Figures/M_samples.pdf')
plt.show()
u_samples = u_samples[0:s, :]
f_samples = f_samples[0:s, :]
# %% Show results. Average over samples
u_samples_store = u_samples
u_samples_box = u_samples
f_samples_box = f_samples

u_mean = torch.mean(u_samples_box, dim=0)
u_upper = u_mean + 1.96*torch.std(u_samples_box, dim=0).data
u_lower = u_mean - 1.96*torch.std(u_samples_box, dim=0).data
f_mean = torch.mean(f_samples_box, dim=0)
f_upper = f_mean + 1.96*torch.std(f_samples_box, dim=0).data
f_lower = f_mean - 1.96*torch.std(f_samples_box, dim=0).data
delta_lower_upper = torch.sum(torch.abs(f_upper-f_lower))/(thold_x*2)

plt.plot(x_interpolate, u_mean.data, 'b', label=r'$E[M]$')
plt.plot(x_interpolate, u_upper.data, '--b', label='95% Confidence Interval')
plt.plot(x_interpolate, u_lower.data, '--b')
plt.title(r'Estimated $E[M]=\hat{u}$ and 95% CI')
plt.legend()
plt.grid()
plt.savefig('./Figures/estimated_mean_cf_'+str(thold_x)+'.pdf')
plt.show()

plt.scatter(x_space, y, 15, 'g', label='Observations')
plt.plot(x_interpolate, f_mean.data, 'r', label=r'$\hat{f}$')
plt.plot(x_interpolate, f_upper.data, '--r', label=r'95%  CI')
plt.plot(x_interpolate, f_lower.data, '--r')
plt.title(r'$\hat{f}$, 95% CI'+' and CI area ='+r'{:.2f}'.format(delta_lower_upper.item()))
plt.legend()
plt.grid()
plt.savefig('./Figures/estimated_f_cf_'+str(thold_x)+'.pdf')
plt.show()



#%% Calculate Kernel mean and Variance
K_tensor = torch.zeros((bnn.N, bnn.N, s))
for i in range(s):
    temp = u_samples_box[i, :]
    K_tensor[:, :, i] = gp.se_kern(temp, bnn.l, bnn.sigma2_f)

#%% plot kernel mean and variance
K_mean = torch.mean(K_tensor, dim=2)
plt.imshow(K_mean.data)
plt.title(r'Mean kernel, $\bar{K}_{**}^{}$. $\Delta=$'+str(2*thold_x))
plt.colorbar()
plt.savefig('./Figures/mean_kernel_'+str(thold_x)+'.pdf')
plt.show()

K_std = torch.std(K_tensor, dim=2)
plt.imshow(K_std.data)
plt.title(r'Standard deviation of sampled kernels, $\sqrt{V[K_{**}^{}]}$. $\Delta=$'+str(2*thold_x))
plt.colorbar()
plt.savefig('./Figures/std_kernel_'+str(thold_x)+'.pdf')
plt.show()

#%% Plot all latent space samples
plt.plot(x_interpolate, u_samples_box.data.t(), 'b', alpha=0.002)
plt.title(r'All samples in $u$-space. $M(X_*^{} ;\theta_s)=U_*^{(s)}$')
plt.grid()
plt.savefig('./Figures/M_samples_'+str(thold_x)+'.pdf')
plt.show()
