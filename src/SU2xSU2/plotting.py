# some additional plotting routines
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

plt.style.use('scientific')
# plt.rcParams.update({'text.usetex': True}) # uncomment when latex is installed


def plot_chain(data_chain_path, ylabel, start_idx=0):
    '''
    Plots raw measurement chain against computer time to gauge when convergence has been achieved.
    Observables of particular interest are the susceptibility (as sufficient burn is needed for the critical slowing down analysis) and 
    the correlation function at a fixed separation (ideally chosen close to the correlation length as this is a slowly converging quantity
    and thus gives a lower bound for burn in).
    If the stored chain has already had some burn in removed, the starting value of the computer time can be adjusted by ``start_idx``. 

    Parameters
    ----------
    data_chain_path: str
        path to measurement chain file (must be .npy)
    ylabel: str
        label for the plotted measurement. When Latex commands are contained, pass as raw string i.e. r'$\chi$'
    start_idx: int (optional)
        amount of burn in already removed from the measurement chain
    '''
    data = np.load(data_chain_path)
    comp_time = np.arange(data.size) + start_idx

    fig = plt.figure()

    plt.plot(comp_time, data)
    plt.xlabel('computer time')
    plt.ylabel(ylabel)

    plt.show()


def correlation_func_plot(data_path, plot_path, fit_upper=None, show_plot=True, ybottom=None, xright=None):
    '''
    Produces a plot (with a logarithmic y axis) of the correlation function data (stored at ``data_path``) and the fitted, analytical expectation. 
    The plot is saved at ``plot_path`` with the fitting range and other plot parameters can be adjusted through the arguments.

    Allows manual adjustment of fitting the correlation length to the processed correlation function data (normalized and mirror averaged).
    A plot with the new fitting is produced and the inferred correlation length, its error and the associated chi2 are printed.
    These can then be manually added to (for example) a data/corlen_data.txt file.

    Parameters
    ----------
    data_path: str
        file path to the correlation function data that will be plotted. The file is assumed to be .npy and 
        contain the rows separation [0,L/2], correlation function, correlation function error respectively 
    plot_path: str 
        file path (including file extension) to store the plot at
    fit_upper: int (optional)
        largest separation (in units of the lattice spacing) to be included in the fit.
        If left as ``None``, only all non-zero values of the correlation function will be included in the fit
    show_plot: bool (optional)
        shows produced plot before saving it
    ybottom: float (optional)
        lower limit for y axis
    xright: float (optional)
        upper limit for x axis
    '''
    def fit(d,xi):
        return (np.cosh((d-L_2)/xi) - 1) / (np.cosh(L_2/xi) - 1)
    
    ds, cor, cor_err = np.load(data_path)
    L_2 = ds[-1]
    
    if fit_upper is not None:
        mask = ds <= fit_upper
    else:
        mask = cor > 0

    popt, pcov = curve_fit(fit, ds[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    fig = plt.figure()

    plt.errorbar(ds, cor, yerr=cor_err, fmt='.', zorder=1)
    ds_fit = np.linspace(0, ds[mask][-1], 500)
    plt.plot(ds_fit, fit(ds_fit,*popt), c='g', zorder=2, label='$\\xi = %.3f \pm %.3f$\n $\chi_r^2 = %.2f$'%(cor_length, cor_length_err, reduced_chi2))
    # set axes limits
    if ybottom is not None:
        plt.ylim(bottom=ybottom, top=2)
    if xright is not None:
        plt.xlim(right=xright)
    plt.yscale('log')
    plt.xlabel(r'wall separation $d$ [$a$]')
    plt.ylabel(r'wall wall correlation $C_{ww}(d)$')
    plt.legend(loc='upper right')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    
    if show_plot:
        plt.show()

    # check if path contains directory. If it does, the directory is created should it not exist already
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close() # for memory purposes

    print('corlen: %.18E \ncorlen_err: %.18E \nchi2: %.18E'%(cor_length, cor_length_err, reduced_chi2))
    return


def internal_energy_density_plot(D, simdata_path, plot_path, show_plot=True):
    '''
    Produces a plot comparing the numerically found internal energy density, stored at ``simdata_path``,
    with the weak and strong coupling expansions (w.c. and s.c.). The plot is saved at ``plot_path``.
    The weak coupling expansion is only plotted for D=2. 

    Parameters
    ----------
    D: int
        dimension of the lattice
    simdata_path: str
        path to .txt file (including extension) containing beta, internal energy density and its error stored as rows
    plot_path: str
        path of plot file (with file extension)
    show_path: bool (optional)
        if True (default), the plot will be shown
    ''' 
    betas, e, e_err = np.loadtxt(simdata_path)

    # strong coupling expansion: eq 2.21 in Guha, Lee, *Improved mean field studies of SU(N) chiral models and comparison with numerical simulations*,
    # `Nucl. Phys. B240, 141 (1984) <https://doi.org/10.1016/0550-3213(84)90473-5>`_
    b_s = np.linspace(0,1)
    strong = 1/2*b_s + 1/12*(3*D-4)*b_s**3 - 1/12*(24*D**2-75*D+52)*b_s**5

    # weak coupling expansion: couldn't not find a closed form expression for general D apart from eq 3.8 in Brihaye, Rossi, 
    # *The weak-coupling phase of lattice spin and gauge models*, `Nucl. phys. B235, 226 (1984) <https://doi.org/10.1016/0550-3213(84)90099-3>`_
    # and p.145-146 in Guha, Lee, *Improved mean field studies of SU(N) chiral models and comparison with numerical simulations*,
    # `Nucl. Phys. B240, 141 (1984) <https://doi.org/10.1016/0550-3213(84)90473-5>`_
    if D == 2:
        Q1 = 0.0958876
        Q2 = -0.0670
        b_w = np.linspace(0.6, 4)
        weak = 1 - 3/(8*b_w) * (1 + 1/(16*b_w) + (1/64 + 3/16*Q1 + 1/8*Q2)/b_w**2)

    fig = plt.figure()
    plt.errorbar(betas, e, yerr=e_err, fmt='.', label='FA HMC')
    plt.plot(b_s, strong, c='b', label='strong coupling')
    if D == 2:
        plt.plot(b_w, weak, c='r', label='weak coupling')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'internal energy density $e$')
    plt.legend(loc='lower right')

    if show_plot:
        plt.show()

    # check if path contains any directories and create those if they don't exist already
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close()

    return


def mass_lambda_plot(simdata_path, plot_path, show_plot=True):
    '''
    Plots the mass over lambda ratio against beta, based on the correlation length data stored at ``simdata_path``.
    The beta function is used at 3 loop accuracy and the integral occurring in the definition of the renormalization scale Lambda
    is evaluated numerically. To assess the convergence of the simulation data to the continuum mass gap prediction, the latter is added to the plot. 

    Parameters
    ----------
    simdata_path: str
        path to .txt file (including extension) containing L, beta, correlation length, its error and the reduced chi-squared stored as rows
    plot_path: str
        path of plot file (with file extension)
    show_path: bool (optional)
        if True (default), the plot will be shown
    '''
    data = np.loadtxt(simdata_path)
    _, betas, xi, xi_err, _ = data

    # beta function coefficients
    N = 2
    b0 = N / (8*np.pi)
    b1 = N**2 / (128*np.pi**2)
    G1 = 0.04616363
    b2 = 1/(2*np.pi)**3 * (N**3)/128 * ( 1 + np.pi*(N**2 - 2)/(2*N**2) - np.pi**2*((2*N**4-13*N**2+18)/(6*N**4) + 4*G1) ) 

    # numerical integration 
    def integrand(x):
        '''
        Integrand in expression for the renormalisation scale using the beta function at 3 loop accuracy.
        
        Parameters
        ----------
        x: float
            value of the coupling constant squared i.e. x=g^2

        Returns
        -------
        inte: float
            the integrand
        '''
        beta_3l = -b0*x**2 - b1*x**3 - b2*x**4  
        inte = 1/beta_3l + 1/(b0*x**2) - b1/(b0**2*x)
        return inte
    
     # Lambda times the lattice spacing a is denoted by the variable F
    F = np.zeros_like(betas)
    pre_factor = (2*np.pi*betas)**(1/2) * np.exp(-2*np.pi*betas)

    for i,beta in enumerate(betas):
        res, err = quad(integrand, 0, 4/(N*beta))
        F[i] = pre_factor[i] * np.exp(-res)

    mass_lambda = 1/xi * 1/F
    mass_lambda_err = mass_lambda / xi * xi_err

    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure()
    plt.errorbar(betas, mass_lambda, yerr=mass_lambda_err, fmt='.', label='FA HMC')
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles=(0, (5,10)), color='k', label='continuum prediction')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$m / \Lambda_{L}$')
    plt.legend(loc='lower right')

    if show_plot:
        plt.show()

    # check if path contains any directories and create those if they don't exist already
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close()

    return


def critical_slowing_plot(unaccel_data_path, accel_data_path, corlengthdata_path, xticks, plot_path, show_plot=True):
    '''
    Plots the integrated autocorrelation time (IAT) of an observable (O) against correlation length for standard and accelerated simulations.
    The data has been previously computed via :py:meth:`SU2xSU2.analysis.critical_slowingdown` and is stored at ``unaccel_data_path`` and ``accel_data_path`` respectively.
    Fitted power laws are superimposed on the data, allowing to quantify the degree of critical slowing down through the fitted value 
    of the critical exponent 'z'. The plot is stored at ``plot_path``.

    Parameters
    ----------
    unaccel_data_path: str
        path to .txt file (including extension) using standard (unaccelerated) HMC.
        The following row structure is assumed: Ls, betas, IATs, IAT error, O, O error, simulation time, acceptance rate
    accel_data_path: str
        path to .txt file (including extension) using accelerated HMC.
        The following row structure is assumed: Ls, betas, IATs, IAT error, O, O error, simulation time, acceptance rate
    corlengthdata_path: str
        path to .txt file containing the correlation length associated to the simulations at beta as its third row.
        Usually, this corresponds to ``corlengthdata_path`` of :py:meth:`SU2xSU2.analysis.mass_lambda`
    xticks: list
        list of int or floats containing location of major ticks for the axis plotting the correlation length
    plot_path: str
        path of plot file (with file extension)
    show_path: bool (optional)
        if True (default), the plot will be shown
    '''
    def power_law(x, z, c):
        return c*x**z

    def linear_func(x, z, b):
        return z*x + b 

    def fit_IAT(xi, IAT, IAT_err):
        '''
        Fits power law for integrated autocorrelation time (IAT) as function of the correlation length xi.

        Parameters
        ----------
        xi: (n,) array
            values of the correlation length for different values of beta
        IAT: (n,) array
            values of the susceptibility IAT for different values of beta

        Returns
        -------
        popt: list length 2
            fitted parameters of the power law
        z: float
            the critical dynamical exponent of xi quantifying the degree of critical slowing down
        z_err: float
            error of the found dynamical exponent 
        '''
        log_IAT = np.log(IAT)
        log_IAT_err = IAT_err / IAT
        popt, pcov = curve_fit(linear_func, np.log(xi), log_IAT, sigma=log_IAT_err, absolute_sigma=True)
        z = popt[0]
        z_err = np.sqrt(pcov[0][0])

        return popt, z, z_err

    xis = np.loadtxt(corlengthdata_path)[2]
    n = len(xis)
    
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc = np.zeros((2,n)), np.zeros((2,n))

    _, _,  IATs[0], IATs_err[0], chis[0], chis_err[0], times[0], acc[0] = np.loadtxt(unaccel_data_path)
    _, _, IATs[1], IATs_err[1], chis[1], chis_err[1], times[1], acc[1] = np.loadtxt(accel_data_path)


    fig = plt.figure()
    cut = None # change to numerical value to define the range of fitting (exclusive)

    # get critical exponent
    fits = np.zeros((2,xis[:cut].shape[0]))
    zs, zs_err, red_chi2s = np.zeros((3,2)) 
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(xis[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(xis[:cut], popt[0], np.exp(popt[1]))
        r = IATs[k][:cut] - fits[k]
        red_chi2s[k] = np.sum((r/IATs[k][:cut])**2) / (fits[k].size - 2) # dof = number of observations - number of fitted parameters

    # critical slowing down plot
    plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='x', markersize=4, label='HMC $z = %.3f \pm %.3f$'%(zs[0],zs_err[0]))
    # plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='x', markersize=4, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    plt.plot(xis[:cut], fits[0], c='b')
    plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='r', fmt='.', label='FA HMC $z = %.3f \pm %.3f$'%(zs[1],zs_err[1]))
    # plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='r', fmt='.', label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[1],zs_err[1], red_chi2s[1]))
    plt.plot(xis[:cut], fits[1], c='r')

    plt.xscale('log')
    # set x ticks manually
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.yscale('log')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'integrated autocorrelation time $\tau_{\chi}$')
    plt.legend()

    if show_plot:
        plt.show()
    
    # check if path contains directory. If it does, the directory is created should it not exist already
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close() # for memory purpose

    return


def cost_function_ratio_plot(unaccel_data_path, accel_data_path, corlengthdata_path, xticks, yticks, plot_path, show_plot=True):
    '''
    Plots the ratio of the const functions for unaccelerated and accelerated simulations. A power law is fitted to the ratio and superimposed in the plot.
    The paths ``unaccel_data_path`` and ``accel_data_path`` contain the simulation data collected in :py:meth:`SU2xSU2.analysis.critical_slowingdown`, specifically
    the simulation time and acceptance rate as well as the integrated autocorrelation time (IAT) of some observable 'O'.

    Parameters
    ----------
    unaccel_data_path: str
        path to .txt file (including extension) using standard (unaccelerated) HMC.
        The following row structure is assumed: Ls, betas, IATs, IAT error, O, O error, simulation time, acceptance rate
    accel_data_path: str
        path to .txt file (including extension) using accelerated HMC.
        The following row structure is assumed: Ls, betas, IATs, IAT error, O, O error, simulation time, acceptance rate
    corlengthdata_path: str
        path to .txt file containing the correlation length associated to the simulations at beta as its third row.
        Usually, this corresponds to ``corlengthdata_path`` of :py:meth:`SU2xSU2.analysis.mass_lambda`
    xticks: list
        list of int or floats containing location of major ticks for the axis plotting the correlation length
    yticks: list
        list of int or floats containing location of major ticks for the axis plotting the cost function ratio
    plot_path: str
        path of plot file (with file extension)
    show_path: bool (optional)
        if True (default), the plot will be shown
    '''
    def linear_func(x, z, b):
        return z*x + b 
    
    xis = np.loadtxt(corlengthdata_path)[2]
    n = len(xis)
    
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc = np.zeros((2,n)), np.zeros((2,n))

    _, _,  IATs[0], IATs_err[0], chis[0], chis_err[0], times[0], acc[0] = np.loadtxt(unaccel_data_path)
    _, _, IATs[1], IATs_err[1], chis[1], chis_err[1], times[1], acc[1] = np.loadtxt(accel_data_path)

    # fit power law to ratio of cost functions
    cost_funcs = times/acc * np.sqrt(IATs)
    cost_funcs_err = cost_funcs * IATs_err/(2*IATs)
    ratio = cost_funcs[0]/cost_funcs[1]
    ratio_err = cost_funcs_err[0]/cost_funcs[1] + cost_funcs[0]/cost_funcs[1]**2 * cost_funcs_err[1]
    log_ratio_err = ratio_err / ratio
    popt, _ = curve_fit(linear_func, np.log(xis), np.log(ratio), sigma=log_ratio_err)
    fit_ratio = np.exp( linear_func(np.log(xis), *popt) )

    fig = plt.figure()

    plt.errorbar(xis, ratio, yerr=ratio_err, fmt='.')
    plt.plot(xis, fit_ratio, c='r', label=r'fitted power law $\sim \xi^{%.3f}$'%popt[0])
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'cost function ratio HMC/FA HMC')
    # set x ticks manually
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks(xticks, minor=False)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    # set y ticks manually
    ax.set_yscale('log')
    ax.set_yticks(yticks, minor=False)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    plt.legend()

    if show_plot:
        plt.show()

    # check if path contains directory. If it does, the directory is created should it not exist already
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close() # for memory purpose
    return


def effective_mass_plot(data_path, xright=None, ytop=None, ybottom=None, show_plot=True, plot_path='plots/effective_mass.pdf'):
    '''
    Produces an effective mass plot using the processed correlation function data stored at ``data_path``.
    The data is expected to contain the separation, the correlation function value and its error row wise.
    The effective mass will be computed based on the assumption that the correlation function follows the shape of
    a cosh as analytically expected due to periodic boundary conditions. 
    As the effective mass becomes noisy for large separations, the plot range can be adjusted using the remaining keywords.  

    Parameters
    ----------
    data_path: str
        path to .npy file containing the averaged (ensemble and across the symmetry axis at L/2) and normalized correlation function
    xright: float
        upper limit of the separation shown in the plot
    ytop: float
        upper limit for the effective mass
    ybottom: float
        lower limit for the effective mass 
    '''

    def effective_mass(cor, cor_err):
        '''
        Computes the effective mass and its error based on a cosh correlation function.
        A lattice of even size is assumed.

        cor: (L/2)
            value of wall to wall correlation function on the first half of the lattice
        cor_err: (L/2)
            error of correlation function on the first half of the lattice

        Returns
        m_eff: (L/2,)
            effective mass
        m_eff_err: (L/2)
            error of the effective mass
        '''
        rel_err = cor_err / cor # relative error
        cor_1, cor_err_1 = np.roll(cor, -1), np.roll(cor_err, -1) # shift to d+1
        rel_err_1 = cor_err_1 / cor_1
        cor__1, cor_err__1 = np.roll(cor, 1), np.roll(cor_err, 1) # shift to d-1
        rel_err__1 = cor_err__1 / cor__1

        A, B = cor_1/cor, cor__1/cor
        x = (A+B)/2 
        m_eff = np.arccosh(x)

        delta_x = 1/2 * (A*(rel_err_1 - rel_err) + B*(rel_err__1 - rel_err))
        # delta_x = A/2*(np.sqrt(rel_err_1**2 + rel_err**2)) + B/2*(np.sqrt(rel_err__1**2 + rel_err**2))
        m_eff_err = np.abs(1/np.sqrt(x**2-1) * delta_x)

        return m_eff, m_eff_err

    data = np.load(data_path)
    ds, cor, cor_err = data[:3]
    m_eff, m_eff_err = effective_mass(cor, cor_err)

    fig = plt.figure()

    plt.errorbar(ds, m_eff, yerr=m_eff_err, fmt='.')
    if xright is not None:
        plt.xlim(right=xright)
    if ybottom is not None:
        plt.ylim(bottom=ybottom)
    if ytop is not None:
        plt.ylim(top=ytop)
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel(r'effective mass $m_{eff}$')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    
    if show_plot:
        plt.show()
    # create directories in plot_path if necessary
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close()