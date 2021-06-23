import torch
import gpytorch


K_module = gpytorch.kernels.RBFKernel()
def se_kern(u_kern, lengthscale, sigma2_f):
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
        out = y_pred.t() @ alpha_pred + B_pred_det
        f_pred = K_pred @ alpha_pred

    return f_pred, out


def gp_uspace(mod, u_pred, y_pred):
    K_pred = se_kern(u_pred, mod.l, mod.sigma2_f)
    B = K_pred + mod.sigma2_n * torch.eye(mod.N)
    alpha_pred, B_LU = torch.solve(y_pred, B)
    yhat = K_pred @ alpha_pred
    return yhat


def NLML(u_pred, y_pred, N, l, sigma2_f, sigma2_n):  # Calculate negative log marginal likelihood (NLML)
    K_pred = se_kern(u_pred, l, sigma2_f)
    B = K_pred + sigma2_n * torch.eye(N)
    alpha_pred, B_LU = torch.solve(y_pred, B)
    _, B_pred_det = torch.linalg.slogdet(B)
    nlml = y_pred.t() @ alpha_pred + B_pred_det
    return nlml


def gp_pred(xobs, xstar, lengthscale, sigma2_f):
    Kobs = se_kern(xobs, lengthscale, sigma2_f)
    K_module.lengthscale = lengthscale*lengthscale
    temp = K_module(xstar, xobs)
    out = sigma2_f*temp.evaluate()
    return out
