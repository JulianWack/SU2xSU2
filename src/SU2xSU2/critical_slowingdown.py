import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import SU2xSU2, get_avg_error
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


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


def chi_IAT_scaling():
    '''
    Computes the IAT of the susceptibility for the beta,N value pairs used in the asymptotic scaling plot.
    The computation is done for accelerated and unaccelerated dynamics to highlight the difference in scaling which is quantified by the fitted value of the 
    critical exponent z. 
    The acceleration mass parameter is chosen as the inverse of the fitted correlation length, which was found to yield close to optimal acceleration.
    '''
    a = 1
    Ns, betas, xis, _, _ = np.loadtxt('data/corlen_beta.txt')
    M = 100000 # number of trajectories to simulate for each N, beta pair
  
    accel_bool = [False, True]
    n = len(betas)
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc = np.zeros((2,n)), np.zeros((2,n))

    chain_path = ['data/slowdown/rawchains/unaccel/','data/slowdown/rawchains/accel/'] # location to store chains. Needed dirs automatically created in SU2xSU2 class
    file_path = ['data/slowdown/unaccel.txt', 'data/slowdown/accel.txt'] # location for final results depending on use of acceleration
    # create needed directories if the dont exist already
    for path in file_path:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

    description = 'dynamics with %d total measurements of susceptibility \nN, beta, IAT and error, chi and error, simulation time [sec], acceptance rate'%M
    des_str = ['Unaccelerated '+description, 'Accelerated '+description]

    prev_ell, prev_eps = [4,4], [1/4, 1/4] 
    for i,beta in enumerate(betas):
        for k, accel in enumerate(accel_bool):
            beta_str = str(np.round(beta, 4)).replace('.', '_')
            model_paras = {'N':Ns[i], 'a':1, 'ell':prev_ell[k], 'eps':prev_eps[k], 'beta':beta, 'mass':1/xis[i]}
            paras_calibrated = calibrate(model_paras, accel=accel)
            prev_ell[k], prev_eps[k] = paras_calibrated['ell'], paras_calibrated['eps']

            model = SU2xSU2(**paras_calibrated)
            rawchain_path = chain_path[k]+'beta_'+beta_str
            sim_paras = {'M':M, 'thin_freq':1, 'burnin_frac':1/50, 'accel':accel, 'measurements':[model.susceptibility], 'chain_paths':[rawchain_path]}
            model.run_HMC(**sim_paras) 

            # get ensemble average and IAT of susceptibility
            data = np.load(rawchain_path+'.npy') # raw chain of susceptibility
            chis[k,i], chis_err[k,i], IATs[k,i], IATs_err[k,i] = get_avg_error(data, get_IAT=True)
            times[k,i], acc[k,i] = model.time, model.acc_rate

            np.savetxt(file_path[k], np.row_stack((Ns, betas, IATs[k], IATs_err[k], chis[k], chis_err[k], times[k], acc[k])), header=des_str[k])
        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)


    # make plot
    cut = None # range to fit

    # get critical exponent
    fits = np.zeros((2,xis[:cut].shape[0]))
    zs, zs_err, red_chi2s = np.zeros((3,2)) 
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(xis[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(xis[:cut], popt[0], np.exp(popt[1]))
        r = IATs[k][:cut] - fits[k]
        red_chi2s[k] = np.sum((r/IATs[k][:cut])**2) / (fits[k].size - 2) # dof = number of observations - number of fitted parameters


    # critical slowing down plot
    fig = plt.figure(figsize=(8,6))
   
    plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    plt.plot(xis[:cut], fits[0], c='b')
    plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='r', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[1],zs_err[1], red_chi2s[1]))
    plt.plot(xis[:cut], fits[1], c='r')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'autocorrelation time $\tau_{\chi}$')
    plt.legend(prop={'size': 12}, frameon=True)

    # plt.show()
    fig.savefig('plots/crit_slowing.pdf')


    # cost function plot
    fig = plt.figure(figsize=(16,6))

    cost_funcs = times/acc * np.sqrt(IATs)
    ratio = cost_funcs[0]/cost_funcs[1]
    popt, _ = curve_fit(linear_func, np.log(xis), np.log(ratio))
    fit_ratio = np.exp( linear_func(np.log(xis), *popt) )

    plt.scatter(xis, ratio, marker='x')
    plt.plot(xis, fit_ratio, c='r', label=r'fitted power law $\sim \xi^{%.3f}$'%popt[0])
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'cost function ratio HMC/FA HMC')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'size': 12}, frameon=True)
    # plt.show()
    fig.savefig('plots/cost_function.pdf')

chi_IAT_scaling()