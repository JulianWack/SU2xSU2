# various plotting routines using stored data
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import SU2xSU2, get_avg_error, corlength
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 30})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])
mpl.rcParams['xtick.major.size'], mpl.rcParams['ytick.major.size'] = 10, 10
mpl.rcParams['xtick.minor.size'], mpl.rcParams['ytick.minor.size'] = 5, 5


def run_sim():
    '''
    Runs a single simulation with hard coded parameters 
    '''
    # # manual parameters
    # model = SU2xSU2(L=64, a=1, ell=7, eps=1/7, beta=0.8667)
    # model.run_HMC(100000, 1, 0, accel=True, measurements=[model.ww_correlation_func], chain_paths=['data/corfunc_long'], partial_save=5000) 

    # calibration
    model_paras = {'L':96, 'a':1, 'ell':15, 'eps':1/15, 'beta':1}
    paras_calibrated = calibrate(model_paras, accel=True)

    model = SU2xSU2(**paras_calibrated)
    file_path = 'data/burn_in_beta1_N96_unaccel'
    sim_paras = {'M':3000, 'thin_freq':1, 'burnin_frac':0, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':[file_path]}
    model.run_HMC(**sim_paras) 

# run_sim()



def corfunc_plot():
    '''
    Produces a plot of the correlation function over the range [0,L) with a linear y scale using a stored raw measurement data
    '''
    def corfunc_plot_linear(ww_cor, ww_cor_err):
        '''
        Plots the correlation function on a linear scale
        '''
        # normalize and use periodic bcs to get correlation for wall separation of L to equal that of separation 0
        ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
        ww_cor, ww_cor_err = np.concatenate((ww_cor, [ww_cor[0]])), np.concatenate((ww_cor_err, [ww_cor_err[0]]))

        ds = np.arange(ww_cor.size)
        
        fig = plt.figure(figsize=(8,6))

        plt.errorbar(ds, ww_cor, yerr=ww_cor_err, fmt='.', capsize=2)
        plt.yscale('log')
        plt.xlabel(r'wall separation [$a$]')
        plt.ylabel('wall-wall correlation')
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
        plt.show()
    
    # load raw chain, get ensemble average
    corfunc_chain = np.load('data/burn_in_beta1_N96_accel.npy')
    cor, cor_err = get_avg_error(corfunc_chain)
    # plot ensemble average of correlation function on linear scale
    corfunc_plot_linear(cor, cor_err)

# corfunc_plot()


def plot_burn_in():
    '''
    Plots raw chain against computer time to gauge when convergence has been achieved.
    observables of interest:
        susceptibility: sufficient burn in needed for critical slowing down plot
        correlation function at a fixed separation (ideally chosen close to the correlation length): slowly converging quantity and thus gives a lower bound for burn in
    
    Note that it is necessary to account for any burn in that has been rejected already before storing the data
    '''
    # adjust manually depending on what chain is examined
    start_idx = 0
    end_idx = 10000
    comp_time = np.arange(start_idx, end_idx)

    # chi
    # full_chain = np.load('data/slowdown/rawchains/unaccel/beta_1_1333.npy')
    # data = full_chain[start_idx:end_idx]
    # y_label = r'$\chi$'

    # correlation function
    sep = 92 # roughly xi
    full_chain = np.load('data/corfuncs/rawchains/1_4.npy')
    data = full_chain[start_idx:end_idx,sep]
    y_label = r'$C_{ww}(\xi)$'
    

    fig = plt.figure(figsize=(8,6))

    plt.plot(comp_time, data)
    plt.xlabel('computer time')
    plt.ylabel(y_label)

    plt.show()

# plot_burn_in()


def chi_vs_xi():
    '''
    Fit a power law relation between chi and xi
    '''
    def power_law(x, z, c):
        return c*x**z

    def linear_func(x, z, b):
        return z*x + b 

    def fit_chi(xi, chi, chi_err):
        '''
        Fit power law for susceptibility as function of the correlation length xi.

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
        log_chi = np.log(chi)
        log_chi_err = chi_err / chi
        popt, pcov = curve_fit(linear_func, np.log(xi), log_chi, sigma=log_chi_err, absolute_sigma=True)

        return popt, pcov

    n = 10
    chi, chi_err = np.zeros((2,n))

    _, _, _, _, chi, chi_err, _ = np.loadtxt('data/slowdown/accel.txt')
    xis = np.loadtxt('data/corlen_beta.txt')[2,:n]

    cut = None
    fit = np.zeros(xis[:cut].shape[0])
    popt, pcov = fit_chi(xis[:cut], chi[:cut], chi_err[:cut])
    C = 1/np.exp(popt[1]) # constant in power law
    fit = power_law(xis[:cut], popt[0], 1/C)
    r = chi[:cut] - fit
    red_chi2 = np.sum((r/chi[:cut])**2) / (fit.size - 2) # dof = number of observations - number of fitted parameters

    err_exponent = np.sqrt(pcov[0,0])
    print('Exponent: %.3f +/- %.3f'%(popt[0], err_exponent))
    err_b = np.sqrt(pcov[1,1])
    print('C: %.3f +/- %.3f'%(C, err_b*C))


    plt.errorbar(xis, chi, yerr=chi_err, fmt='.', capsize=2)
    plt.plot(xis[:cut], fit, c='b')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'susceptibility $\chi$')
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend(prop={'size':12}, frameon=True)
    plt.show()

# chi_vs_xi()


def sim_time_compare():
    '''
    Plots the ratio of the HMC and FA HMC simulation time  
    '''
    n = 10
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times = np.zeros((2,n))

    _, _,  IATs[0], IATs_err[0], chis[0], chis_err[0], times[0] = np.loadtxt('data/slowdown/unaccel.txt')
    _, _, IATs[1], IATs_err[1], chis[1], chis_err[1], times[1] = np.loadtxt('data/slowdown/accel.txt')

    xis = np.loadtxt('data/corlen_beta.txt')[2,:n]

    fig = plt.figure(figsize=(12,6))

    plt.scatter(xis, times[0]/times[1], marker='x')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'simulation time ratio HMC/FA HMC')
    plt.xscale('log')
    plt.show()

# sim_time_compare()


def couping_expansion():
    betas, e_avg, e_err = np.loadtxt('data/coupling_expansion.txt')

    # make strong and weak coupling expansion plot
    b_s = np.linspace(0,1)
    strong = 1/2*b_s + 1/6*b_s**3 + 1/6*b_s**5

    Q1 = 0.0958876
    Q2 = -0.0670
    b_w = np.linspace(0.6, 4)
    weak = 1 - 3/(8*b_w) * (1 + 1/(16*b_w) + (1/64 + 3/16*Q1 + 1/8*Q2)/b_w**2)

    fig = plt.figure(figsize=(16,9))

    plt.errorbar(betas, e_avg, yerr=e_err*10, fmt='.', capsize=2, label='FA HMC')
    plt.plot(b_s, strong, c='b', label='strong coupling')
    plt.plot(b_w, weak, c='r', label='weak coupling')
    plt.xlabel(r'$\beta$')
    plt.ylabel('internal energy density $e$')
    plt.legend(prop={'size': 22}, frameon=True, loc='lower right')

    plt.show()
    # fig.savefig('plots/coupling_expansion.pdf')

# couping_expansion()


def coupling_exp_residual():
    '''plots residuals of couling expansion to make error bars visible'''
    def strong_func(b):
        return 1/2*b + 1/6*b**3 + 1/6*b**5

    def weak_func(b):
        Q1 = 0.0958876
        Q2 = -0.0670
        return 1 - 3/(8*b) * (1 + 1/(16*b) + (1/64 + 3/16*Q1 + 1/8*Q2)/b**2)
    
    betas, e_avg, e_err = np.loadtxt('data/coupling_expansion.txt')

    strong_mask = betas<0.7
    b_strong = betas[strong_mask]
    weak_mask = betas>0.7
    b_weak = betas[weak_mask]

    strong = strong_func(b_strong)
    weak = weak_func(b_weak)

    fig = plt.figure(figsize=(16,6))

    # small beta value close to zero such that data/expansion blows up
    plt.errorbar(b_strong, (strong-e_avg[strong_mask])/strong, yerr=e_err[strong_mask]/strong, c='b', fmt='x', capsize=2, label='(s.c. - data)/s.c.')
    plt.errorbar(b_weak, (weak-e_avg[weak_mask])/weak, yerr=e_err[weak_mask]/weak, c='r', fmt='x', capsize=2, label='(w.c. - data)/w.c.')
    plt.xlabel(r'$\beta$')
    plt.ylabel('normalised residual')
    plt.legend(prop={'size': 12}, frameon=True)

    plt.show()

# coupling_exp_residual()


def chi_speed():
    '''
    Makes a ratio plot of the CPU time needed to compute the susceptibility via a double sum and the cross correlation theorem.
    '''
    L, CC, DS = np.loadtxt('data\chi_speed.txt')

    fig = plt.figure(figsize=(16,6))

    # plt.scatter(L, DS/CC, marker='x')
    plt.plot(L, DS, c='b', label='naive sum')
    plt.plot(L, CC, c='r', label='FFT approach')

    plt.xlabel('lattice size $L$')
    plt.ylabel('CPU time [sec]')
    # plt.ylabel('CPU time ratio double sum / cross correlation')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks(np.logspace(-3,4,8))
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.legend(prop={'size':12}, frameon=True)

    plt.show()

# chi_speed()


def accelmass_search():
    # load data
    masses, cost_func, cost_func_err, times, acc_rates, chi_IATs, chi_IATs_err = np.loadtxt('../SU2xSU2_preproduction/data/mass_parameter.txt')

    # normalise to M=1/xi
    cost_func_normed = cost_func/cost_func[-1]
    cost_func_normed_err = cost_func_err/cost_func[-1] 

    fig = plt.figure(figsize=(16,9))

    plt.errorbar(masses[:-1], cost_func_normed[:-1], yerr=cost_func_normed_err[:-1], fmt='.', capsize=2)
    plt.errorbar(masses[-1], cost_func_normed[-1], yerr=cost_func_normed_err[-1], c='r', fmt='.', capsize=2, label=r'$M=1/\xi$')
    plt.xlabel('acceleration mass $M$')
    plt.ylabel(r'cost function normalised to $M=1/\xi$')
    plt.legend(prop={'size':22}, frameon=True)
    # plt.yscale('log')
    plt.show()

# accelmass_search()



### Main result plots ###
def correlation_func_plot():
    '''Allows manual adjustment of fitting the correlation length to the processed correlation function data (normalized and mirror averaged).
    A plot with the new fitting is produced and the inferred correlation length, its error and the associated chi2 are printed.
    These can then be manually added to the data/corlen_data.txt file.
    '''
    def fit(d,xi):
        return (np.cosh((d-L_2)/xi) - 1) / (np.cosh(L_2/xi) - 1)
    
    ds, cor, cor_err = np.load('data/corfuncs/beta_0_8667.npy')

    fit_upper = 27 # inclusive upper bound on wall separation to include in fitting
    L_2 = ds[-1]
    
    mask = ds <= fit_upper
    # mask = cor > 0
    popt, pcov = curve_fit(fit, ds[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    fig = plt.figure(figsize=(16,9))

    plt.errorbar(ds, cor, yerr=cor_err, fmt='.', capsize=2, zorder=1)
    ds_fit = np.linspace(0, ds[mask][-1], 500)
    plt.plot(ds_fit, fit(ds_fit,*popt), c='g', zorder=2, label='$\\xi = %.3f \pm %.3f$\n $\chi_r^2 = %.2f$'%(cor_length, cor_length_err, reduced_chi2))
    plt.ylim(bottom=2e-4, top=2)
    plt.xlim(right=33)
    plt.yscale('log')
    plt.xlabel(r'wall separation $d$ [$a$]')
    plt.ylabel('wall wall correlation $C_{ww}(d)$')
    plt.legend(prop={'size':22}, frameon=True, loc='upper right') # location to not conflict with error bars
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.show()

    print('corlen: %.18E \ncorlen_err: %.18E \nchi2: %.18E'%(cor_length, cor_length_err, reduced_chi2))
    
# correlation_func_plot()


def asym_scaling_plot():
    '''Makes asymptotic scaling plot using the beta function at either 2 or 3 loop accuracy.'''
    data = np.loadtxt('data/corlen_beta.txt')
    _, betas, xi, xi_err, _ = data

    def Lambda(betas, loops=3):
        '''Returns the lattice spacing a times the Lambda parameter at 2 or 3 loop accuracy in the lattice scheme.'''
        N = 2
        b0 = N / (8*np.pi)
        b1 = N**2 / (128*np.pi**2) 

        # 2 loop result
        res = (2*np.pi*betas)**(1/2) * np.exp(-2*np.pi*betas)
        if loops == 2:
            return res

        # 3 loop correction
        G1 = 0.04616363
        b2 = 1/(2*np.pi)**3 * (N**3)/128 * ( 1 + np.pi*(N**2 - 2)/(2*N**2) - np.pi**2*((2*N**4-13*N**2+18)/(6*N**4) + 4*G1) )
        correction =  (b1**2 - b0*b2)/(N*b0**3)*4/betas
        res = res * (1 + correction)

        return res

    mass_lambda = 1/xi * 1/Lambda(betas)
    mass_lambda_err = mass_lambda / xi * xi_err

    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure(figsize=(16,9))
    plt.errorbar(betas, mass_lambda, yerr=mass_lambda_err, fmt='.', capsize=2, label='FA HMC')
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles='--', color='k', label='continuum prediction')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$m / \Lambda_{L}$')
    plt.legend(prop={'size':22}, frameon=True, loc='lower right')

    plt.show()

# asym_scaling_plot()


def asym_scaling_E_scheme():
    ''' Compares mass over lambda inferred using the standard and by rescaling it through the internal energy density.
    Introduced as E scheme in Rossi, Vicari (https://link.aps.org/doi/10.1103/PhysRevD.49.1621) and has shown great improvement for N>=3 SU(N)xSU(N) models.
    
    Below is sum data from this paper which allows to reproduce their fig.3
    # N=3
    # betas = np.array([0.18, 0.225, 0.25, 0.27, 0.27, 0.27, 0.29, 0.30, 0.315])
    # E = np.array([0.74118, 0.62819, 0.55589, 0.49992, 0.50000, 0.50003, 0.45172, 0.43111, 0.40400])
    # xi = np.array([1.003, 1.87, 3.027, 4.79, 4.78, 4.81, 7.99, 10.41, 15.5])
    '''
    N = 2 # SU(N) group order
    data = np.loadtxt('data/corlen_beta.txt')
    _, betas, xi, xi_err, _ = data
    _, e, e_err = np.loadtxt('data/E_scheme.txt')
    E = 1-e
    E_err = e_err

    def Lambda(betas, n, loops=3):
        '''Returns the lattice spacing a times the Lambda parameter at 2 or 3 loop accuracy in the lattice scheme.
        Due to the different normalization of the action, a parameter n is needed such that for
        standard coupling: n = 2 
        e scheme: n = 8
        '''
        N = 2
        b0 = N / (8*np.pi)
        b1 = N**2 / (128*np.pi**2) 

        # 2 loop result
        res = (n*np.pi*betas)**(1/2) * np.exp(-n*np.pi*betas)
        if loops == 2:
            return res

        # 3 loop correction
        G1 = 0.04616363
        b2 = 1/(2*np.pi)**3 * (N**3)/128 * ( 1 + np.pi*(N**2 - 2)/(2*N**2) - np.pi**2*((2*N**4-13*N**2+18)/(6*N**4) + 4*G1) )
        correction =  (b1**2 - b0*b2)/(N*b0**3)* (8/n)/betas
        res = res * (1 + correction)

        return res
    
    # standard coupling
    ML = 1/xi * 1/Lambda(betas, 2)
    ML_err = ML / xi * xi_err

    # rescaled coupling
    betas_E = (N**2-1)/(8*N**2*E)
    betas_E_err = betas_E * E_err/E
    ML_E = 1/xi * np.exp(np.pi*(N**2-2)/(4*N**2)) * 1/Lambda(betas_E, 8)
    ML_E_err = ML_E/xi * xi_err + ML_E*8*np.pi*(1-1/(16*np.pi*betas_E))*betas_E_err # ignores error from beta_E as E small and E_err of order 1e-4

    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure(figsize=(16,9))

    plt.errorbar(betas, ML, yerr=ML_err, c='b', fmt='.', capsize=2, label='standard coupling')
    plt.errorbar(betas, ML_E, yerr=ML_E_err, c='r', fmt='.', capsize=2, label='rescaled coupling')
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles='--', color='k', label='continuum prediction')
    # plt.ylim(bottom=20, top=100)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$M / \Lambda_{L,2l}$')
    plt.legend(prop={'size': 18}, frameon=True, loc='lower right')

    plt.show()

# asym_scaling_E_scheme()


def critslowing_plot():
    '''Makes critical slowing down plot'''
    def power_law(x, z, c):
        return c*x**z

    def linear_func(x, z, b):
        return z*x + b 

    def fit_IAT(xi, IAT, IAT_err):
        '''
        Fit power law for integrated autocorrelation time as function of the correlation length xi.

        Returns
        popt: list length 2
            optimal parameters of fitted function
        z: float
            the dynamical exponent of xi
        z_err: float
            error of the found dynamical exponent 
        '''
        log_IAT = np.log(IAT)
        log_IAT_err = IAT_err / IAT
        popt, pcov = curve_fit(linear_func, np.log(xi), log_IAT, sigma=log_IAT_err, absolute_sigma=True)
        z = popt[0]
        z_err = np.sqrt(pcov[0][0])

        return popt, z, z_err

    n = 13 # number of xi measurements at different beta
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc = np.zeros((2,n)), np.zeros((2,n))

    _, _,  IATs[0], IATs_err[0], chis[0], chis_err[0], times[0], acc[0] = np.loadtxt('data/slowdown/unaccel.txt')
    _, _, IATs[1], IATs_err[1], chis[1], chis_err[1], times[1], acc[1] = np.loadtxt('data/slowdown/accel.txt')

    xis = np.loadtxt('data/corlen_beta.txt')[2,:n]

    cut = None # exclusive upper bound of fit

    # get critical exponent
    fits = np.zeros((2,xis[:cut].shape[0]))
    zs, zs_err, red_chi2s = np.zeros((3,2)) 
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(xis[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(xis[:cut], popt[0], np.exp(popt[1]))
        r = IATs[k][:cut] - fits[k]
        red_chi2s[k] = np.sum((r/IATs[k][:cut])**2) / (fits[k].size - 2) # dof = number of observations - number of fitted parameters


    # # critical slowing down plot
    fig = plt.figure(figsize=(16,9))

    # # move one data serires slightly for better visibility
    # # plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='g', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    # # plt.errorbar(xis+0.5, IATs[0], yerr=IATs_err[0], c='g', fmt='.', capsize=2)
    
    # plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='x', markersize=4, capsize=2, label='HMC $z = %.3f \pm %.3f$'%(zs[0],zs_err[0]))
    plt.plot(xis[:cut], fits[0], c='b')
    # plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='r', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[1],zs_err[1], red_chi2s[1]))
    plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='r', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$'%(zs[1],zs_err[1]))
    plt.plot(xis[:cut], fits[1], c='r')

    # # add data points to test finite size effects
    # # fs_data = np.loadtxt('data/mixed_data/sd_finite_size_test.txt')
    # # plt.errorbar(xis[-3:], fs_data[3], yerr=fs_data[4], c='b', alpha=0.25, fmt='.', capsize=2, label='HMC half the lattice size')


    plt.xscale('log')
    # set x ticks manually
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.yscale('log')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'integrated autocorrelation time $\tau_{\chi}$')
    plt.legend(prop={'size': 22}, frameon=True)
    plt.show()


    # cost function plot
    cost_funcs = times/acc * np.sqrt(IATs)
    ratio = cost_funcs[0]/cost_funcs[1]
    popt, _ = curve_fit(linear_func, np.log(xis), np.log(ratio))
    fit_ratio = np.exp( linear_func(np.log(xis), *popt) )

    fig = plt.figure(figsize=(16,9))

    plt.scatter(xis, ratio, marker='x')
    plt.plot(xis, fit_ratio, c='r', label=r'fitted power law $\sim \xi^{%.3f}$'%popt[0])
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'cost function ratio HMC/FA HMC')
    # plt.xscale('log')
    # set x ticks manually
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.yscale('log')
    plt.legend(prop={'size': 22}, frameon=True)
    plt.show()

# critslowing_plot()