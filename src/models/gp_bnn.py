# %% Libraries
import numpy as np
import torch
from src.models import gp_nn_class
import matplotlib.pyplot as plt
from src.models import gp_bnn_class
import gpytorch
from src.models import function_generator as fg

torch.pi = torch.acos(torch.zeros(1)).item() * 2
K_module = gpytorch.kernels.RBFKernel()

# %% Generate Data
bnn = gp_bnn_class.BNN()  # generate BNN class
# set bnn hyper-parameters
bnn.sigma2_likelihood = 1; bnn.l = torch.sqrt(torch.tensor(1.0))
bnn.N = 200; bnn.sigma2_f = 1; bnn.sigma2_n = torch.tensor(0.0005)
bnn.sigma2_prior = 1; bnn.get_total_no_param()
bnn.prior()  # Generate weights/biases from the normal prior.

# Generate Data
delta = 1  # threshold of zero information. delta \in (0,5)
x_space = torch.cat((torch.linspace(-5, -delta, 100), torch.linspace(delta, 5, 100)))
x_space = torch.reshape(x_space, (bnn.N, 1))
y = fg.square_func(threshold=0, x=x_space, amplitude=1, n=bnn.N, sigma2_n=bnn.sigma2_n)

# x_space = torch.cat((torch.linspace(-5, -delta, 100), torch.linspace(delta, 5, 100)))
# x_space = torch.reshape(x_space, (bnn.N, 1))
# y, y_true, x_true = fg.wall_pulse_func(1, 0.5, bnn.N, bnn.sigma2_n)
# plt.scatter(x_space, y)
# plt.show()
# plt.plot(x_true, y_true)
# plt.show()

# generate X_*
x_star = torch.reshape(torch.linspace(torch.min(x_space), torch.max(x_space), 200), (200, 1))

# %% Sampling with warm start
T = 10000  # no. of HMCMC iterations
s = 0  # no. samples counter
e = 0  # no. of exploration stages counter
L = 5  # no. of leapfrog steps at each HCMC iteration
alt_flag = True  # if true then turn on posterior that use the marginal likelihood p(y|u)
M = int(T / 50)  # no. of cycles for epsilon
beta = 0.2  # Take beta proportion of each cyclic to use for exploration only
ep_space, t_burn, poly, cyclic = fg.ep_generate(T, M, ep0=0.0005, ep_max=0.001, ep_min=0.000002,
                                                gamma=0.99, t_burn=500, ep_type="Cyclic")  # Generate cyclic step-size

# HMCMC. Warm start using neural network minimizing negative log marginal likelihood (NLML)
net = gp_nn_class.NN()
net.get_total_no_param()
net.l = bnn.l; net.N = bnn.N; net.sigma2_f = bnn.sigma2_f; net.sigma2_n = bnn.sigma2_n  # Same hyper-parameters as the BNN
net.train_nn(x_space=x_space, y=y, EPOCHS=4000, BATCH_SIZE=50)  # Train M for warm start
theta = net.get_param()  # Get the warm start parameters
bnn.ws_u = net.forward(x_space)  # Save warm start to use for neural network likelihood

# init energies and gradients
bnn.update_param(theta)
u = bnn.forward(x_space)
U_nn = bnn.energy_nn(u, theta)
_, U_gp = bnn.energy_gp(u, y, alt_flag)
U = U_gp + U_nn
grad_U = bnn.grad_calc(U)

# Init tensors to store samples
u_samples = torch.zeros((T, bnn.N))  # u-space samples
f_samples = torch.zeros((T, bnn.N))  # Prediction from each u sample. E[f_* |X_*, X, y, theta]
v_samples = torch.zeros((T, bnn.N))  # Variance on prediction for each sample V[f_* |X_*, X, y, theta]
u_samples_p = torch.zeros((T, bnn.N))
G = torch.zeros(T)  # \gradE_U(theta) Gradient L2-norm for each t.
U_nn_p = np.nan; U_gp_p = np.nan; up = np.nan; u_cum = torch.zeros((bnn.N, 1))

for t in range(T):
    rp = torch.normal(torch.zeros(bnn.total_no_param), torch.ones(bnn.total_no_param))
    r = rp
    theta_p = theta
    grad_U_p = grad_U
    U_p = U

    # Leapfrog
    ep = ep_space[t]  # fetch current step-size
    for i in range(L):
        rp = rp - ep * grad_U_p * 0.5  # first half step momentum
        theta_p = theta_p + ep * rp  # full step position/parameter

        # Calculate gradient of U_p (\gradE_U(theta'))
        bnn.update_param(theta_p)
        up = bnn.forward(x_space)  # Proposed u-space
        U_nn_p = bnn.energy_nn(up, theta_p)  # bnn potential energy
        fp, U_gp_p = bnn.energy_gp(up, y, alt_flag)  # gp potential energy
        U_p = U_gp_p + U_nn_p  # Proposed Pot. Energy
        grad_U_p = bnn.grad_calc(U_p)  # \grad E_U(theta')
        rp = rp - ep * grad_U_p * 0.5  # second half step momentum

    G[t] = torch.sqrt(torch.sum(grad_U_p ** 2)) / bnn.total_no_param  # Norm of Gradient
    if (torch.fmod(torch.tensor(t - 1), t_burn) / t_burn < beta and cyclic) or (t < t_burn and poly):
        #  exploration stage
        e += 1  # exploration step counter
        theta = theta_p; bnn.update_param(theta)  # update parameters
        U_nn = U_nn_p  # neural network potential energy update
        U_gp = U_gp_p  # gp potential energy update
        U = U_p  # potential energy update
    else:
        #  sampling stage
        K = torch.sum(r ** 2) / 2  # kinetic energy
        Kp = torch.sum(rp ** 2) / 2  # proposed kinetic energy
        alpha = -U_p - Kp + U + K  # acceptance probability in log-space
        if torch.log(torch.rand(1)) < alpha:
            # accept theta' as a sample from p(theta | X y)
            theta = theta_p; bnn.update_param(theta)  # Accept proposal theta_p
            U_nn = U_nn_p  # neural network potential energy update
            U_gp = U_gp_p  # gp potential Energy update
            U = U_p  # potential energy Update
            u_star = bnn.forward(x_star)  # predict u-space.
            u_samples[s, :] = u_star.flatten()  # Store each u-space prediction
            plt.plot(x_star, u_star.data, 'b', alpha=0.02)  # plot each u_star

            f_star, V = bnn.gp_pred_var(up, u_star, y)  # prediction mean and variance
            f_samples[s, :] = f_star
            v_samples[s, :] = V
            u_samples_p[s, :] = up.flatten()
            s = s + 1  # no. of samples accepted counter
        else:
            # No sample. Keep current position/parameter
            bnn.update_param(theta)
    print(t, ' : ', s, ' : ', e)  # Print progress

# plot samples from u-space
plt.xlabel(r'$x$')
plt.ylabel(r'$M(x)=u$')
plt.title(r'Each sample, $M_i$, of the probabilistic mapping $M$')
plt.grid()
plt.savefig('./Figures/M_samples' + str(delta) + '.pdf')
plt.show()

# keep only the samples
u_samples = u_samples[0:s, :]
f_samples = f_samples[0:s, :]
v_samples = v_samples[0:s, :]
u_samples_p = u_samples_p[0:s, :]

# %% Calculate mean and variance of samples. And plot results
f_mean = torch.mean(f_samples, dim=0)  # 1/S sum_{s=1}^S f_star[s]
u_mean = torch.mean(u_samples_p, dim=0)  # E[U | X, y] approximation

# CI1
f_upper_CI1 = f_mean + 1.96 * torch.sqrt(torch.var(f_samples, dim=0) + bnn.sigma2_n)
f_lower_CI1 = f_mean - 1.96 * torch.sqrt(torch.var(f_samples, dim=0) + bnn.sigma2_n)
CI1_area = torch.sum(torch.abs(f_upper_CI1 - f_lower_CI1)) / (delta * 2)

f_upper_CI2 = torch.mean(f_samples + 1.96 * torch.sqrt(v_samples + bnn.sigma2_n), dim=0)
f_lower_CI2 = torch.mean(f_samples - 1.96 * torch.sqrt(v_samples + bnn.sigma2_n), dim=0)
CI2_area = torch.sum(torch.abs(f_upper_CI2 - f_lower_CI2)) / (delta * 2)  # CI area

# f_hat +/- CI1
plt.scatter(x_space, y, 15, 'g', label='Observations')
plt.plot(x_star, f_mean.data, 'r', label=r'$\hat{f}$')
plt.plot(x_star, f_upper_CI1.data, '--r', label=r'95%  CI1')
plt.plot(x_star, f_lower_CI1.data, '--r')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.title(r'$\hat{f}$, 95% CI on samples' + ' and CI area =' + r'{:.2f}'.format(CI1_area.item()))
plt.legend()
plt.grid()
plt.savefig('./Figures/estimated_f_cf_sample_' + str(delta) + '.pdf')
plt.show()

# f_hat +/- CI2
plt.scatter(x_space, y, 15, 'g', label='Observations')
plt.plot(x_star, f_mean.data, 'r', label=r'$\hat{f}$')
plt.plot(x_star, f_upper_CI2.data, '--r', label=r'95%  CI2')
plt.plot(x_star, f_lower_CI2.data, '--r')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.title(r'$\hat{f}$, 95% CI' + ' and CI area =' + r'{:.2f}'.format(CI2_area.item()))
plt.legend()
plt.grid()
plt.savefig('./Figures/estimated_f_cf_' + str(delta) + '.pdf')
plt.show()

# %% Calculate Kernel mean and Variance

K_tensor = torch.zeros((bnn.N, bnn.N, s))  # Store all kernels for each theta_s
for i in range(s):
    temp = u_samples[i, :]
    K_tensor[:, :, i] = bnn.se_kern(temp, temp)

# Calculate kernel mean and variance
K_mean = torch.mean(K_tensor, dim=2)  # \bar{K}_star,star
K_std = torch.std(K_tensor, dim=2)  # \sqrt{V[K_star,star]}

# plot kernel mean and variance
plt.imshow(K_mean.data)
plt.title(r'Mean kernel, $\bar{K}_{**}^{}$. $\Delta=$' + str(2 * delta))
plt.colorbar()
plt.savefig('./Figures/mean_kernel_' + str(delta) + '.pdf')
plt.show()
plt.imshow(K_std.data)
plt.title(r'Standard deviation of sampled kernels, $\sqrt{V[K_{**}^{}]}$. $\Delta=$' + str(2 * delta))
plt.colorbar()
plt.savefig('./Figures/std_kernel_' + str(delta) + '.pdf')
plt.show()

# %% Manifold Gaussian Process

net.train_nn(x_space=x_space, y=y, EPOCHS=10000, BATCH_SIZE=50)  # Train M for warm start
theta = net.get_param()  # Get the warm start parameters

#%%

u_space = net.forward(x_space)  # Save warm start to use for neural network likelihood
u_star_nn = net.forward(x_star)
f_star_nn, V = bnn.gp_pred_var(u_space, u_star_nn, y)  # prediction mean and variance
plt.plot(x_star, u_star_nn.data)
plt.show()
plt.scatter(x_space, y)
plt.plot(x_star, f_star_nn, 'r')
plt.plot(x_star, f_star_nn.data + 1.96 * torch.sqrt(V.data + bnn.sigma2_n), '--r')
plt.plot(x_star, f_star_nn.data - 1.96 * torch.sqrt(V.data + bnn.sigma2_n), '--r')
plt.grid()
plt.show()

#%%
def fitFunc(x_star, u_obs, u_star, y):
    f_star, V = bnn.gp_pred_var(u_obs, u_star, y)  # prediction mean and variance
    plt.plot(x_star, u_star.data)
    plt.show()
    #plt.scatter(x_space, y)
    plt.plot(x_star, f_star.data, 'r')
    plt.plot(x_star, f_star.data + 1.96 * torch.sqrt(V.data + bnn.sigma2_n), '--r')
    plt.plot(x_star, f_star.data - 1.96 * torch.sqrt(V.data + bnn.sigma2_n), '--r')
    plt.grid()
    plt.show()

sample = 2
u_test = u_samples[sample, :]
u_obs = u_samples_p[sample, :]

fitFunc(x_star, u_obs, u_test, y)




# %% Plot all latent space samples

# plt.plot(x_star, u_samples.data.t(), 'b', alpha=0.002)
# plt.title(r'All samples in $u$-space. $M(X_*^{} ;\theta_s)=U_*^{(s)}$')
# plt.grid()
# plt.savefig('./Figures/M_samples_'+str(delta)+'.pdf')
# plt.show()