
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.colors as mcolors
from numpy import emath as npe


# Adapted directly from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in tqdm(enumerate(costs), desc="Pareto Front, Points Evaluated"):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def fit_power_law(x_vals, y_vals, fix_c0=None, negative_power_term=False, suboptimality=False, optimal_value=None, x_space_len=50, p0=None, maxfev=20000, verbose=False):
    """
    NOTE DEPRECATED
    See power_scaling_fit below
    """
    assert not suboptimality or optimal_value is not None
    assert not negative_power_term or not suboptimality

    def powerlaw(x, c0, c, m):
        return c0 + (c / x)**m
    
    def fixed_powerlaw(x, c, m):
        return fix_c0 + (c / x)**m
    
    def subopt_powerlaw(x, c0, c, m):
        return ((c0 / optimal_value) - 1) + (1 / optimal_value) * (c / x)**m

    def fixed_subopt_powerlaw(x, c, m):
        return ((fix_c0 / optimal_value) - 1) + (1 / optimal_value) * (c / x)**m
    
    def negative_powerlaw(x, c0, c, m):
        return c0 - (c / x)**m
    
    def fixed_negative_powerlaw(x, c, m):
        return fix_c0 - (c / x)**m
    
    if suboptimality:
        power_func = fixed_subopt_powerlaw if fix_c0 is not None else subopt_powerlaw
    elif negative_power_term:  # negative powerlaw supports upper asymptote, which makes more sense with [0, 1] bound in rand-opt-norm
        power_func = fixed_negative_powerlaw if fix_c0 is not None else negative_powerlaw
    else:
        power_func = fixed_powerlaw if fix_c0 is not None else powerlaw

    popt, pcov = curve_fit(power_func, x_vals, y_vals, p0=p0, maxfev=maxfev)

    if verbose:
        print("Power fit parameters:")
        print(*popt)

    x_space = np.geomspace(min(x_vals), max(x_vals), num=x_space_len)
    power_fit = power_func(x_space, *popt)

    return x_space, power_fit, popt, lambda x: power_func(x, *popt)


def power_plot(plt_func, x, y, line_col, idxs=None, geom_start=0.001, geom_end=110, y_fit_amplify=None, negative_power_term=False, fix_c0=None):
    if idxs is not None:
        x = x[idxs]
        y = y[idxs]

    yfit = y_fit_amplify * y if y_fit_amplify is not None else y  # y_fit_amplify helps with very small ys which can mess with fit
    fix_c0 = y_fit_amplify * fix_c0 if y_fit_amplify is not None and fix_c0 is not None else fix_c0

    _, _, popt, pf = fit_power_law(x, yfit, negative_power_term=negative_power_term, fix_c0=fix_c0)

    if fix_c0 is not None:
        c, m = popt
        c0 = fix_c0
    else:
        c0, c, m = popt

    x_geom = np.geomspace(geom_start, geom_end)
    yfit = pf(x_geom)

    if y_fit_amplify is not None:
        yfit /= y_fit_amplify
        c0 /= y_fit_amplify
        c *= (y_fit_amplify ** (1/m))

    sign_chr = "-" if negative_power_term else "+"

    plt_func(x_geom, yfit, ':', color=line_col, label=f"$Scaling \; Law = {c0:.3f} {sign_chr} (\\frac{{x}}{{{{{c:.2f}}}}})^{{{-m:.3f}}}$")


def power_scaling_fit(x_vals, y_vals, x_bounds, c0="fit", c1="fit", mode="grow", sign="positive", rescale=True, p0=None, maxfev=200000, verbose=True, near_linear=False):
    """
    Generic power law fitting
    
    c0: y-intercept (keep as 'fit' to tune, otherwise fix at 0 or another float value)
    c1: x-offset (ditto)
    mode: 'grow' or 'decay' (positive or negative exponent 'm', respectively)
    sign: change to 'negative' to flip scaling over horizontal axis c0 (grow in negative direction, or decay upwards toward asymptote)
    """
    assert mode in ("grow", "decay")
    assert sign in ("positive", "negative")

    if rescale:  # rescale x and y centered around 1 for better fit with scipy.curve_fit, see https://stackoverflow.com/questions/75098813/using-scipy-optimize-curve-fit-to-find-parameters-of-a-curve-and-getting-covari
        x_scale = np.median(x_vals)
        y_scale = np.median(y_vals)
    else:
        x_scale = 1
        y_scale = 1

    def fixed_powerlaw(x, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0 + s * (((x / x_scale) - c1) / c_f)**m_f)

    def c0_flex_powerlaw(x, c0_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (((x / x_scale) - c1) / c_f)**m_f)

    def c1_flex_powerlaw(x, c1_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0 + s * (((x / x_scale) - c1_f) / c_f)**m_f)

    def full_flex_powerlaw(x, c0_f, c1_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (((x / x_scale) - c1_f) / c_f)**m_f)
    
    epsilon = 1e-8

    if c0 == "fit":
        c0_low = -np.inf
        c0_high = np.inf

    if c1 == "fit":
        c1_low = -np.inf
        c1_high = 0.999 * (np.concatenate([x_vals, x_bounds]).min() / x_scale)  # ensures x power term can't become negative

    c_low = epsilon  # ensures c term can't become negative (or divide by 0)
    c_high = np.inf

    if mode == "grow" and near_linear:
        m_low = 0.995
        m_high = 1.005
    elif mode == "grow":
        m_low = epsilon
        m_high = np.inf
    elif mode == "decay":
        m_low = -np.inf
        m_high = -epsilon

    if c0 == "fit" and c1 == "fit":
        powerlaw = full_flex_powerlaw
        bounds = ([c0_low, c1_low, c_low, m_low],
                  [c0_high, c1_high, c_high, m_high])
    elif c0 == "fit":
        powerlaw = c0_flex_powerlaw
        bounds = ([c0_low, c_low, m_low],
                  [c0_high, c_high, m_high])
    elif c1 == "fit":
        powerlaw = c1_flex_powerlaw
        bounds = ([c1_low, c_low, m_low],
                  [c1_high, c_high, m_high])
    else:
        powerlaw = fixed_powerlaw
        bounds = ([c_low, m_low],
                  [c_high, m_high])     
            
    popt, pcov = curve_fit(powerlaw, x_vals, y_vals, p0=p0, bounds=bounds, maxfev=maxfev, method="trf")

    if verbose:
        print(f"popt : {popt} | x_scale {x_scale}, y_scale {y_scale}")

    pfn = lambda x: powerlaw(x, *popt)
    
    x_min, x_max = x_bounds
    x_fit = np.geomspace(x_min, x_max, num=500)

    y_fit = pfn(x_fit)

    if c0 == "fit" and c1 != "fit":
        c0_out, c_out, m_out = popt
        popt = c0_out, c1, c_out, m_out
    elif c1 == "fit" and c0 != "fit":
        c1_out, c_out, m_out = popt
        popt = c0, c1_out, c_out, m_out
    if c0 != "fit" and c1 != "fit":
        c_out, m_out = popt
        popt = c0, c1, c_out, m_out

    return x_fit, y_fit, (popt, x_scale, y_scale)


def save_fit_eq(file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale, b=None):
    """
    If fit is: y = y_scale * (c0 + (((x / x_scale) - c1) / c)**m)
    This converts to: y = c0_pure + ((x - c1_pure) / c_pure)**m

    Omits c0 and c1 if near 0, and supports negative m (flipping fraction if negative)

    """
    assert b is None  # API match with expfit for convenience

    epsilon = 1e-10

    c0_pure = y_scale * c0
    c1_pure = x_scale * c1
    c_pure = (x_scale * c) / (y_scale**(1/m))

    c0_term = f"{c0_pure} " if np.abs(c0) > epsilon else ""
    sign_term = "- " if sign == "negative" else ("+ " if np.abs(c0) > epsilon else "")
    c1_term = f" - {c1_pure}" if np.abs(c1) > epsilon else ""
    frac_term = f"({c_pure} / (x{c1_term}))"  if m < 0 else f"((x{c1_term}) / {c_pure})" 

    out_line = f"{id_str} : y = {c0_term}{sign_term}{frac_term}^{np.abs(m)}\n"

    with open(file_str, "a") as f:
        f.write(out_line)


def clear_fits(file_str):
    with open(file_str, "w") as f:
        pass


def exp_scaling_fit(x_vals, y_vals, x_bounds, c0="fit", c1="fit", mode="grow", sign="positive", rescale=True, p0=None, maxfev=200000, verbose=True, near_linear=False):
    """
    Generic exponential law fitting
    
    c0: y-intercept (keep as 'fit' to tune, otherwise fix at 0 or another float value)
    c1: x-offset (ditto)
    mode: 'grow' or 'decay' (positive or negative growth rate base 'm', respectively)
    sign: change to 'negative' to flip scaling over horizontal axis c0 (grow in negative direction, or decay upwards toward asymptote)
    """
    assert not near_linear  # just included for API compatability with power_scaling_fit as a convenience

    assert mode in ("grow", "decay")
    assert sign in ("positive", "negative")

    if rescale:  # rescale x and y centered around 1 for better fit with scipy.curve_fit, see https://stackoverflow.com/questions/75098813/using-scipy-optimize-curve-fit-to-find-parameters-of-a-curve-and-getting-covari
        x_scale = np.median(x_vals)
        y_scale = np.median(y_vals)
    else:
        x_scale = 1
        y_scale = 1

    b = 1.0 if mode == "grow" else -1.0

    def fixed_explaw(x, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0 + s * (m_f**((x / x_scale) - c1) / c_f)**b)

    def c0_flex_explaw(x, c0_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (m_f**((x / x_scale) - c1) / c_f)**b)

    def c1_flex_explaw(x, c1_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0 + s * (m_f**((x / x_scale) - c1_f) / c_f)**b)

    def full_flex_explaw(x, c0_f, c1_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (m_f**((x / x_scale) - c1_f) / c_f)**b)
    
    epsilon = 1e-8

    if c0 == "fit":
        c0_low = -np.inf
        c0_high = np.inf

    if c1 == "fit":
        c1_low = -np.inf
        c1_high = np.inf    #0.999 * (np.concatenate([x_vals, x_bounds]).min() / x_scale)  # ensures x power term can't become negative

    c_low = epsilon  # ensures c term can't become negative (or divide by 0)
    c_high = np.inf

    m_low = epsilon  # ensures base can't become negative (or divide by 0)
    m_high = np.inf

    if c0 == "fit" and c1 == "fit":
        explaw = full_flex_explaw
        bounds = ([c0_low, c1_low, c_low, m_low],
                  [c0_high, c1_high, c_high, m_high])
    elif c0 == "fit":
        explaw = c0_flex_explaw
        bounds = ([c0_low, c_low, m_low],
                  [c0_high, c_high, m_high])
    elif c1 == "fit":
        explaw = c1_flex_explaw
        bounds = ([c1_low, c_low, m_low],
                  [c1_high, c_high, m_high])
    else:
        explaw = fixed_explaw
        bounds = ([c_low, m_low],
                  [c_high, m_high])     
            
    popt, pcov = curve_fit(explaw, x_vals, y_vals, p0=p0, bounds=bounds, maxfev=maxfev, method="trf")

    if verbose:
        print(f"popt : {popt} | x_scale {x_scale}, y_scale {y_scale}")

    efn = lambda x: explaw(x, *popt)
    
    x_min, x_max = x_bounds
    x_fit = np.linspace(x_min, x_max, num=500)

    y_fit = efn(x_fit)

    if c0 == "fit" and c1 != "fit":
        c0_out, c_out, m_out = popt
        popt = c0_out, c1, c_out, m_out
    elif c1 == "fit" and c0 != "fit":
        c1_out, c_out, m_out = popt
        popt = c0, c1_out, c_out, m_out
    if c0 != "fit" and c1 != "fit":
        c_out, m_out = popt
        popt = c0, c1, c_out, m_out

    return x_fit, y_fit, (popt, x_scale, y_scale)


def save_expfit_eq(file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale, b):
    """
    If fit is: y = y_scale * (c0 + s * (m**((x / x_scale) - c1) / c)**b)
    This converts to: y = c0_pure + (m_pure**(x - c1_pure) / c_pure)**b

    Omits c0 and c1 if near 0
    b in (-1, 1)
    """
    epsilon = 1e-10
    assert np.abs(b) - epsilon < 1 and np.abs(b) + epsilon > 1

    c0_pure = y_scale * c0
    c1_pure = x_scale * c1
    c_pure = c / (y_scale**(1/b))
    m_pure = m**(1 / x_scale)

    c0_term = f"{c0_pure} " if np.abs(c0) > epsilon else ""
    sign_term = "- " if sign == "negative" else ("+ " if np.abs(c0) > epsilon else "")
    c1_term = f" - {c1_pure}" if np.abs(c1) > epsilon else ""
    frac_term = f"({c_pure} / {m_pure}^(x{c1_term}))"  if b < 0 else f"({m_pure}^(x{c1_term}) / {c_pure})" 

    out_line = f"{id_str} : y = {c0_term}{sign_term}{frac_term}\n"

    with open(file_str, "a") as f:
        f.write(out_line)


def quasi_scaling_fit(x_vals, y_vals, x_bounds, c0="fit", c1="fit", mode="grow", sign="positive", rescale=True, p0=None, maxfev=200000, verbose=True, near_linear=False):
    """
    Quasi-polynomial law fitting of form y = c0 +- x^log_b(x) which is faster than polynomial/power and slower than exponential
    
    c0: y-intercept (keep as 'fit' to tune, otherwise fix at 0 or another float value)
    c1: x-offset (ditto)
    mode: 'grow' or 'decay'
    sign: change to 'negative' to flip scaling over horizontal axis c0 (grow in negative direction, or decay upwards toward asymptote)
    """
    assert not near_linear  # just included for API compatability with power_scaling_fit as a convenience
    assert c1 != "fit"  # not adding gamma support
    assert c0 == "fit"  # requiring beta support

    assert mode in ("grow", "decay")
    assert sign in ("positive", "negative")

    if rescale:  # only y rescaling --> x scaling makes things hard to simplify here
        y_scale = np.median(y_vals)
    else:
        y_scale = 1

    b = 1.0 if mode == "grow" else -1.0

    def quasilaw(x, c0_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (x**(npe.logn(m_f, x)) / c_f)**b)
    
    epsilon = 1e-8

    c0_low = -np.inf
    c0_high = np.inf

    c_low = epsilon  # ensures c term can't become negative (or divide by 0)
    c_high = np.inf

    m_low = 1.0 + epsilon  # ensures log base is strictly >1
    m_high = np.inf

    bounds = ([c0_low, c_low, m_low],
              [c0_high, c_high, m_high])

    if p0 is None:
        p0 = [1.0, 1.0, np.e]  # starting log base at 1 results in a non-function
            
    popt, pcov = curve_fit(quasilaw, x_vals, y_vals, p0=p0, bounds=bounds, maxfev=maxfev, method="trf")

    if verbose:
        print(f"popt : {popt} | x_scale {1}, y_scale {y_scale}")

    qfn = lambda x: quasilaw(x, *popt)
    
    x_min, x_max = x_bounds
    x_fit = np.linspace(x_min, x_max)

    y_fit = qfn(x_fit)

    c0_out, c_out, m_out = popt
    popt = c0_out, c1, c_out, m_out

    return x_fit, y_fit, (popt, 1, y_scale)


def save_quasifit_eq(file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale, b):
    """
    If fit is: y = y_scale * (c0_f + s * (x**(npe.logn(m_f, x)) / c_f)**b)
    This converts to: y = c0_pure + (x**(npe.logn(m_pure, x)) / c_pure)**b)
    
    c1 here for API compatibility but not used
    b in (-1, 1)
    """
    assert x_scale == 1
    epsilon = 1e-10
    assert np.abs(b) - epsilon < 1 and np.abs(b) + epsilon > 1
    assert c1 - epsilon < 0 and c1 + epsilon > 0

    c0_pure = y_scale * c0
    c_pure = c / (y_scale**(1/b))
    m_pure = m

    c0_term = f"{c0_pure} " if np.abs(c0) > epsilon else ""
    sign_term = "- " if sign == "negative" else ("+ " if np.abs(c0) > epsilon else "")
    frac_term = f"({c_pure} / x^(log_{m_pure}(x))"  if b < 0 else f"(x^(log_{m_pure}(x) / {c_pure})" 

    out_line = f"{id_str} : y = {c0_term}{sign_term}{frac_term}\n"

    with open(file_str, "a") as f:
        f.write(out_line)


def subexp_scaling_fit(x_vals, y_vals, x_bounds, c0="fit", c1="fit", mode="grow", sign="positive", rescale=True, p0=None, maxfev=200000, verbose=True, near_linear=False):
    """
    Subexponential fitting of form y = c0 +- m^(x**(1/c1)) which is faster than quasipolynomial and still slower than exponential
    
    c0: y-intercept (keep as 'fit' to tune, otherwise fix at 0 or another float value)
    c1: subexpoenent (0, 1]
    mode: 'grow' or 'decay'
    sign: change to 'negative' to flip scaling over horizontal axis c0 (grow in negative direction, or decay upwards toward asymptote)
    """
    assert not near_linear  # just included for API compatability with power_scaling_fit as a convenience
    assert c1 == "fit"  # subexponent (not gamma)
    assert c0 == "fit"  # requiring beta support

    assert mode in ("grow", "decay")
    assert sign in ("positive", "negative")

    if rescale:  # only y rescaling --> x scaling makes things hard to simplify here
        y_scale = np.median(y_vals)
    else:
        y_scale = 1

    b = 1.0 if mode == "grow" else -1.0

    def subexplaw(x, c0_f, c1_f, c_f, m_f):
        s = 1.0 if sign == "positive" else -1.0
        return y_scale * (c0_f + s * (m_f**(x**(1/c1_f)) / c_f)**b)
    
    epsilon = 1e-8

    c0_low = -np.inf
    c0_high = np.inf

    c1_low = 1.0
    c1_high = np.inf

    c_low = epsilon  # ensures c term can't become negative (or divide by 0)
    c_high = np.inf

    m_low = epsilon  # ensures base can't become negative (or divide by 0)
    m_high = np.inf
        
    bounds = ([c0_low, c1_low, c_low, m_low],
                [c0_high, c1_high, c_high, m_high])

    # if p0 is None:
    #     p0 = [1.0, 2.0, 1.0, 1.0]  # starting with sqrt subexponentiality
            
    popt, pcov = curve_fit(subexplaw, x_vals, y_vals, p0=p0, bounds=bounds, maxfev=maxfev, method="trf")

    if verbose:
        print(f"popt : {popt} | x_scale {1}, y_scale {y_scale}")

    sefn = lambda x: subexplaw(x, *popt)
    
    x_min, x_max = x_bounds
    x_fit = np.linspace(x_min, x_max, num=500)

    y_fit = sefn(x_fit)

    return x_fit, y_fit, (popt, 1, y_scale)


def save_subexpfit_eq(file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale, b):
    """
    If fit is: y = y_scale * (c0 + s * (m**(x**(1/c1) / c)**b)
    This converts to: y = c0_pure + (m_pure**(x**(1/c1_pure) / c_pure)**b

    b in (-1, 1)
    """
    assert x_scale == 1
    epsilon = 1e-10
    assert np.abs(b) - epsilon < 1 and np.abs(b) + epsilon > 1

    c0_pure = y_scale * c0
    c1_pure = c1
    c_pure = c / (y_scale**(1/b))
    m_pure = m

    c0_term = f"{c0_pure} " if np.abs(c0) > epsilon else ""
    sign_term = "- " if sign == "negative" else ("+ " if np.abs(c0) > epsilon else "")
    c1_term = f"^(1/{c1_pure})"
    frac_term = f"({c_pure} / {m_pure}^(x{c1_term}))" if b < 0 else f"({m_pure}^(x{c1_term}) / {c_pure})" 

    out_line = f"{id_str} : y = {c0_term}{sign_term}{frac_term}\n"

    with open(file_str, "a") as f:
        f.write(out_line)


def numpify_metrics(*metric_lists):
    return (np.asarray([metric.value for metric in met_list]) for met_list in metric_lists)


def numpify_steps(*metric_lists):
    return (np.asarray([metric.step for metric in met_list]) for met_list in metric_lists)


def get_sol_eval(client, exp_name, x_param_key, y_metric_key):
    plot_runs = get_all_mlf_runs(client, exp_name)  # NOTE assumes one run per (cost, x) eval

    x_vals = []
    y_vals = []

    for run in plot_runs:
        x_vals.append(int(run.data.params[x_param_key]))
        y_vals.append(client.get_metric_history(run.info.run_id, y_metric_key)[0].value)

    sort_idx = np.argsort(x_vals)

    return np.asarray(y_vals)[sort_idx], np.asarray(x_vals)[sort_idx]


def get_all_mlf_runs(client, exp_name):
    input_exp = client.get_experiment_by_name(exp_name)
    assert input_exp is not None, f"Input experiment '{exp_name}' does not exist"

    return client.search_runs(experiment_ids=[input_exp.experiment_id])  # WARNING this will be too slow if experiment is too large to view via MLflow UI


def get_bind_eval(client, exp_name, run_name, metric_key):  # assumes steps is scale
    bind_runs = get_all_mlf_runs(client, exp_name)
    match_runs = list(filter(lambda r: r.info.run_name == run_name, bind_runs))
    assert len(match_runs) == 1
    bind_run = match_runs[0]

    metrics = client.get_metric_history(bind_run.info.run_id, metric_key)

    x_vals = next(numpify_steps(metrics))
    y_vals = next(numpify_metrics(metrics))

    sort_idx = np.argsort(x_vals)

    return y_vals[sort_idx], x_vals[sort_idx]


def save_fig(fig, format, title, xlabel, ylabel, legend_loc, legend_size):
    ax = fig.gca()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc, prop={'size': legend_size})

    if format in ("eps", "svg"):
        fig.savefig(f"{title}.{format}", format=format)
    else:  # assuming non-vector with dpi
        fig.savefig(f"{title}.{format}", format=format, dpi=300)


def evenly_weighted_average(data, k, mode="valid"):
    """
    Thanks ChatGPT-4
    """
    if data.size < k:
        raise ValueError("The size of the input data array must be greater than or equal to K.")
    
    weights = np.ones(k) / k
    result = np.convolve(data, weights, mode=mode)

    return result


def forceAspect(ax, aspect=1):
    """
    ax.set_aspect('equal') is buggy
    Adapted from https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio
    """
    #aspect is width/height
    yscale_str = ax.get_yaxis().get_scale()
    xscale_str = ax.get_xaxis().get_scale()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    def scale_transform(scale_str, minv, maxv):
        if scale_str == "linear":
            return minv, maxv
        elif scale_str == "log":
            return np.log10(minv), np.log10(maxv)
        
    xmint, xmaxt = scale_transform(xscale_str, xmin, xmax)
    ymint, ymaxt = scale_transform(yscale_str, ymin, ymax)

    asp = abs((xmaxt - xmint) / (ymaxt - ymint)) / aspect

    ax.set_aspect(asp)


def partial_cmap(cmap, minval, maxval):
    """
    Adapted from
    https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar
    """
    return mcolors.LinearSegmentedColormap.from_list(f"trunc({cmap.name},{minval},{maxval})", cmap(np.linspace(minval, maxval, cmap.N)))
