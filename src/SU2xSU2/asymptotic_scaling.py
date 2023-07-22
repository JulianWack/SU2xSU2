import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import SU2xSU2, get_avg_error, corlength
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def mass_lambda():
    '''
    Computes the mass over lambda parameter ratio for a range of beta and L (lattice size along one dimension) value pairs.
    To do so, the correlation length is needed. For each of the considered value pairs, the raw measurement chain for the correlation function is saved
    to memory. A plot of the averaged (ensemble and across the symmetry axis at L/2) and normalized correlation function is produced and saved.
    A plot of mass over lambda ratio is produced, to assess the convergence of the simulation data to the continuum mass gap prediction as beta gets large. 
    '''
    a = 1
    Ns = [40, 40, 64, 64, 64, 96, 96, 160, 160, 224, 400, 512, 700]
    betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4])
    
    xi, xi_err, reduced_chi2 = np.zeros((3,betas.shape[0]))
    prev_ell, prev_eps = 4, 1/4 
    for i,beta in enumerate(betas):
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        model_paras = {'N':Ns[i], 'a':1, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=True)
        prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps']

        model = SU2xSU2(**paras_calibrated)
        file_path = 'data/corfuncs/rawchains/'+beta_str
        sim_paras = {'M':100000, 'thin_freq':1, 'burnin_frac':1/50, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':[file_path]}
        model.run_HMC(**sim_paras) 

        # get ensemble average and find correlation length
        data = np.load(file_path+'.npy') # raw chain of wall wall correlation function
        cor, cor_err = get_avg_error(data) # ensemble avg and err
        file_path = 'data/corfuncs/beta_'+beta_str
        plot_path = 'plots/corfuncs/beta_%s.pdf'%beta_str
        xi[i], xi_err[i], reduced_chi2[i] = corlength(cor, cor_err, file_path, plot_path, make_plot=True, show_plot=False)
        
        des_str = 'correlation lengths inferred from %d measurements of the correlation function for different N and beta pairs: N, beta, xi, xi_err, chi-square per degree of freedom.'%model.M
        np.savetxt('data/corlen_beta.txt', np.row_stack((Ns, betas, xi, xi_err, reduced_chi2)), header=des_str)
        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)


    # make plot
    data = np.loadtxt('data/corlen_beta.txt')
    _, betas, xi, xi_err, _ = data

    mass_lambda = 1/xi * np.exp(2*np.pi*betas) / np.sqrt(2*np.pi*betas)
    mass_lambda_err = mass_lambda / xi * xi_err
    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure(figsize=(8,6))
    plt.errorbar(betas, mass_lambda, yerr=mass_lambda_err, fmt='.', capsize=2)
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles='--', color='k')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$M / \Lambda_{L,2l}$')

    # plt.show()
    plt.savefig('plots/asym_scaling.pdf')


mass_lambda()