import torch
from torch import nn
from torch.autograd import grad
import gpytorch

class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_fi = 1
        self.L1_fo = 32
        self.L2_fi = self.L1_fo
        self.L2_fo = 32
        self.L3_fi = self.L2_fo
        self.L3_fo = 1

        self.L1 = nn.Linear(self.L1_fi, self.L1_fo)
        self.A1 = nn.Tanh()
        self.L2 = nn.Linear(self.L2_fi, self.L2_fo)
        self.A2 = nn.Tanh()
        self.L3 = nn.Linear(self.L3_fi, self.L3_fo, bias=False)

        # Hyper-parameters
        self.sigma2_likelihood = 0
        self.total_no_param = 0
        self.l = torch.sqrt(torch.tensor(0.0))
        self.N = 0
        self.sigma2_f = 0
        self.sigma2_n = torch.tensor(0)
        self.sigma2_prior = 0
        self.K_module = gpytorch.kernels.RBFKernel()

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
        # theta_param is a vector consisting of all neural network parameter
        # Load theta_param into the neural network parameter
        theta_dict = self.state_dict()
        c = 0
        # Construct a parameter dictionary from theta_param
        for param_tensor in theta_dict:
            s = theta_dict[param_tensor].size()  # Get size of current weight matrix
            n = s.numel()  # number of elements
            param = torch.reshape(theta_param[c:c + n], s)  # reshape the part of theta_param
            c = c + n  # index to monitor how far we are in theta_param
            theta_dict[param_tensor] = param  # store in the parameter dictionary
        self.load_state_dict(theta_dict)  # update parameter dictionary

    # calculate dE_U(theta)/dtheta
    def grad_calc(self, energy):
        self.zero_grad()  # reset gradients
        temp = grad(energy, self.parameters())  # dE_U(theta)/dtheta stores as dictionary
        grads = []
        # convert temp into vector
        for g in temp:
            grads.append(g.view(-1))
        grads = torch.cat(grads)
        return grads

    # Get current parameters as a vector
    def get_param(self):
        theta_dict = self.state_dict()  # Parameter dictionary
        out = []  # Parameter vector init
        for param_tensor in theta_dict:
            temp = theta_dict[param_tensor]
            out.append(temp.view(-1))
        out = torch.cat(out)
        return out

    def energy_gp(self, x, y, alt_flag):
        # Calculates:
        #   -potential energy E_U(theta)
        #       by evaluating the numerator of the
        #       log posterior log(p(theta | x,y))
        #   -f_hat= E[f |theta,x,y]
        # atl_flag switch between two posteriors.

        u_pred = self.forward(x)
        K_pred = self.se_kern(u_pred, self.l, self.sigma2_f)
        B = K_pred + self.sigma2_n * torch.eye(self.N)

        # Posterior that use p(y | f)p(f|x)
        if not alt_flag:
            alpha_pred, B_LU = torch.solve(y, B)
            f_hat = K_pred @ alpha_pred  # E[f | x, y, theta]
            ll_gp = torch.sum((y-f_hat) ** 2) / self.sigma2_n  # log(p(y | f))
            _, K_pred_det = torch.linalg.slogdet(K_pred + 1e-4*torch.eye(self.N))
            lp_gp = f_hat.t() @ alpha_pred + K_pred_det  # log(p(f | x))
            U_gp = lp_gp + ll_gp
        else:  # posterior that use p(y | u)
            alpha_pred, B_LU = torch.solve(y, B)
            _, B_pred_det = torch.linalg.slogdet(B)
            U_gp = y.t() @ alpha_pred + B_pred_det  # p(y | x)
            f_hat = K_pred @ alpha_pred
            return f_hat, U_gp

    def se_kern(self, u_kern, l, sigma2_f):
        self.K_module.l = l*l
        temp = self.K_module(u_kern)
        out = sigma2_f*temp.evaluate()
        return out

    def gp_pred(self, x_obs, x_star, yobs):
        # Perform prediction f_hat = E[f_* | x_*, x, y]
        self.K_module.l = self.l*self.l
        temp = self.K_module(x_star, x_obs)
        K_star_obs = self.sigma2_f*temp.evaluate()
        temp = self.K_module(x_obs, x_obs)
        K_obs_obs = self.sigma2_f*temp.evaluate()

        B = K_obs_obs+self.sigma2_n*torch.eye(self.N)
        alpha_pred, B_LU = torch.solve(yobs, B)
        f_hat = K_star_obs @ alpha_pred
        return f_hat

