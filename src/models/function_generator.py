import torch


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
        for i in range(T):
            k = (ep_min/ep_max)**(1/gamma)
            b = k*T/(1-k)
            a = ep_max*b**gamma
            ep_space[i] = a*(b+i)**(-gamma)
    return ep_space, t_burn, poly, cyclic


def wall_pulse_func(fc, a, N, sigma2_n):
    t_space = torch.linspace(0, 10, N)
    out = torch.exp(-a*t_space**2)*torch.exp(1j*2*torch.pi*fc*t_space)
    out = torch.reshape(out.real, (N, 1))
    out = out + torch.flip(out, [0, 1])
    eta = torch.normal(torch.zeros(N, 1), torch.ones(N, 1)*torch.sqrt(sigma2_n))
    return out+eta, out, torch.reshape(t_space, (N, 1))