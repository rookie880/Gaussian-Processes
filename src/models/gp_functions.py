import torch
import gpytorch


K_module = gpytorch.kernels.RBFKernel()


def se_kern(u_kern, lengthscale, sigma2_f):
    K_module.lengthscale = lengthscale*lengthscale
    temp = K_module(u_kern)
    out = sigma2_f*temp.evaluate()
    return out


def gp_prior(model, u_prior):
    K_prior = se_kern(u_prior, model.l, model.sigma2_f)
    B = K_prior + model.sigma2_n * torch.eye(model.N)
    f_prior = torch.distributions.MultivariateNormal(torch.zeros(model.N), B)
    f_prior = f_prior.sample()
    eta = torch.normal(torch.zeros(model.N), torch.ones(model.N) * torch.sqrt(model.sigma2_n))
    out = f_prior + eta
    out = torch.reshape(out, (model.N, 1))
    return out


def gp(model, x, y, alt_flag):
    # Calculates:
    #   -potential energy E_U(theta)
    #       by evaluating the numerator of the
    #       log posterior log(p(theta | x,y))
    #   -f_hat= E[f |theta,x,y]
    # atl_flag switch between two posteriors.

    u_pred = model.forward(x)
    K_pred = se_kern(u_pred, model.l, model.sigma2_f)
    B = K_pred + model.sigma2_n * torch.eye(model.N)

    # Posterior that use p(y | f)p(f|x)
    if not alt_flag:
        alpha_pred, B_LU = torch.solve(y, B)
        f_hat = K_pred @ alpha_pred  # E[f | x, y, theta]
        ll_gp = torch.sum((y-f_hat) ** 2) / model.sigma2_n  # log(p(y | f))
        _, K_pred_det = torch.linalg.slogdet(K_pred + 1e-4*torch.eye(model.N))
        lp_gp = f_hat.t() @ alpha_pred + K_pred_det  # log(p(f | x))
        U_gp = lp_gp + ll_gp
    else:  # posterior that use p(y | u)
        alpha_pred, B_LU = torch.solve(y, B)
        _, B_pred_det = torch.linalg.slogdet(B)
        U_gp = y.t() @ alpha_pred + B_pred_det  # p(y | x)
        f_hat = K_pred @ alpha_pred
    return f_hat, U_gp


#def gp_uspace(model, u_pred, y):
#    K_pred = se_kern(u_pred, model.l, model.sigma2_f)
#    B = K_pred + model.sigma2_n * torch.eye(model.N)
#    alpha_pred, B_LU = torch.solve(y, B)
#    yhat = K_pred @ alpha_pred
#    return yhat


def NLML(u_pred, y, N, l, sigma2_f, sigma2_n):
    # Calculate negative log marginal likelihood (NLML)
    K_pred = se_kern(u_pred, l, sigma2_f)
    B = K_pred + sigma2_n * torch.eye(N)
    alpha_pred, _ = torch.solve(y, B)
    _, B_pred_det = torch.linalg.slogdet(B)
    nlml = y.t() @ alpha_pred + B_pred_det
    return nlml


def gp_pred(xobs, xstar, yobs, lengthscale, sigma2_f, sigma2_n, N):
    # Perform prediction f_hat = E[f_* | x_*, x, y]
    K_module.lengthscale = lengthscale*lengthscale
    temp = K_module(xstar, xobs)
    K_star_obs = sigma2_f*temp.evaluate()
    temp = K_module(xobs, xobs)
    K_obs_obs = sigma2_f*temp.evaluate()

    B = K_obs_obs+sigma2_n*torch.eye(N)
    alpha_pred, B_LU = torch.solve(yobs, B)
    f_hat = K_star_obs @ alpha_pred
    return f_hat

