# Routines to analyse simulation data and to make the main analysis plots
import os
import time
from datetime import timedelta
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.style.use('scientific')
# plt.rcParams.update({'text.usetex': True}) # uncomment when latex is installed

from .SU2xSU2 import SU2xSU2
from .calibrate_paras import calibrate
from .correlations import autocorrelator
from . import plotting


def get_avg_error(data, get_IAT=False):
    '''
    Performs the ensemble average based on a sequence of measurements of a single observable and estimates its error 
    as the standard error on the mean (SEM), corrected by the square root of the observable's integrated autocorrelation time.
    Optionally also returns integrated autocorrelation time and its error.
    Each element of data is refereed to as a data point and can either be a float or a 1D array of length N. 
    The returned average and error will be of the same shape as the data point.

    Parameters
    ----------
    data: (M,) or (M,N) array
        set of measurements made during the chain
    get_IAT: bool (optional)
        False by default. Set to True to return IAT and its error

    Returns
    -------
    avg: float or (N, )
        ensemble average of data
    error: float or (N, )
        autocorrelation corrected SEM
    IAT: float (if ``get_IAT==True``)
        integrated autocorrelation time (IAT)
    IAT_err: float (if ``get_IAT==True``)
        error of IAT
    '''
    if len(data.shape) == 1:
        data = data.reshape((data.shape[0], 1))
    M, N = data.shape # number of repetitions, length of each data point

    avg = np.mean(data, axis=0)
    error = np.std(data, axis=0) / np.sqrt(M)

    # correct error through IAT
    IAT, IAT_err = np.zeros((2,N))
    for i in range(N):
        ts, ACF, ACF_err, IAT[i], IAT_err[i], comp_time = autocorrelator(data[:,i])
        error[i] *= np.sqrt(IAT[i])

    # to return floats if data points were floats
    if N == 1:
        avg, error = avg[0], error[0]
        IAT, IAT_err = IAT[0], IAT_err[0]

    if get_IAT:
        return avg, error, IAT, IAT_err
    
    return avg, error


def get_corlength(ww_cor, ww_cor_err, data_save_path):
    ''' 
    Infers the correlation length based on the ensemble averaged wall to wall correlation function and its error. 
    The correlation function data is processed by normalizing and averaging it about its symmetry axis at L/2 which will be saved
    at ``data_save_path``.
    
    Parameters
    ----------
    ww_cor: (L,) array
        ensemble average of correlation function on a lattice of length L
    ww_cor_err: (L,) array
        associated error for all possible wall separations
    data_save_path: str
        relative path to cwd at which the wall separations, the processed correlation function and its error will be stored 
        row wise as an .npy file (extension not needed in path)

    Returns
    -------
    cor_length: float
        fitted correlation length in units of the lattice spacing
    cor_length_err: float
        error in the fitted correlation length
    reduced_chi2: float
        chi-square per degree of freedom as a goodness of fit proxy
    '''
    def fit(d,xi):
        return np.cosh((d-L_2)/xi) / np.cosh(L_2/xi)

    # normalize and use periodic bcs to get correlation for wall separation of L to equal that of separation 0
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    ww_cor, ww_cor_err = np.concatenate((ww_cor, [ww_cor[0]])), np.concatenate((ww_cor_err, [ww_cor_err[0]]))

    # use symmetry about L/2 due to periodic bcs and mirror the data to reduce errors (effectively increasing number of data points by factor of 2)
    L_2 = int(ww_cor.shape[0]/2) 
    ds = np.arange(L_2+1) # wall separations covering half the lattice length
    cor = 1/2 * (ww_cor[:L_2+1] + ww_cor[L_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:L_2+1]**2 + ww_cor_err[L_2::-1]**2) / np.sqrt(2)

    # store processed correlation function data
    # check if path contains directory. If it does, the directory is created should it not exist already
    dir_path = os.path.dirname(data_save_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    np.save(data_save_path, np.row_stack([ds, cor, cor_err]))
    
    # perform the fit  
    mask = cor > 0 # fitting range
    popt, pcov = curve_fit(fit, ds[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    return cor_length, cor_length_err, reduced_chi2


def internal_energy_coupling_exp(betas, D, Ls, num_traj, burnin_frac, accel=True,
                                chaindata_pathbase='data/energy_density/', simdata_path='data/energy_density.txt', plot_path='plots/coupling_expansion.pdf'):
    '''
    Computes and stores the internal energy per site for the passed value pairs of ``betas``, ``Ls`` for a D-dimensional lattice.
    As a density is computed, the value of the lattice size is not crucial and finite size effects are often negligible.
    For each value pair, ``num_traj`` trajectories are simulated, with ``burnin_frac`` specifying the fraction of these rejected as burn in. 
    By default the Fourier accelerated Hybrid Monte Carlo algorithm is used. For each simulation, the chain of internal energy measurements is
    stored at ``chaindata_pathbase`` and files are labeled by the used value of beta and L. 
    The ensemble average values of the internal energy densities are stored at ``simdata_path`` as a text file.
    A plot, stored at ``plot_path``, is produced comparing the simulation result with the weak and strong coupling expansions (w.c. and s.c.).
    
    Parameters
    ----------
    betas: (n,) array
        values of the model parameter beta for which simulations are performed in ascending order 
    D: int
        dimension of the lattice
    Ls: int or (n,) array
        size of the lattice along one dimension for each simulation at different beta. 
        When an integer is passed, the size will be assumed for all values of beta.
    num_traj: int
        number of trajectories in each simulation
    burnin_frac: float
        fraction of trajectories discarded as burn in
    accel: bool (optional)
        using Fourier Acceleration by default
    chaindata_pathbase: str
        path of the directory where the internal energy measurement chains for all value pairs of beta, L will be saved
    simdata_path: str (optional)
        path of .txt file (relative to the current working directory) to store the ensemble averaged simulation results for the
        internal energy density and its error (with file extension) as well as ``betas`` row-wise
    plot_path: str
        path of final plot file (with file extension)
    '''
    if isinstance(Ls, int):
        Ls = np.full(betas.shape, Ls, dtype='int')

    # set up storage path and add readme file explaining the contents of the directory
    dir_path = os.path.dirname(chaindata_pathbase)
    os.makedirs(dir_path, exist_ok=True)
    with open(chaindata_pathbase + 'readme.txt', 'w') as text_file:
        text_file.write('The internal energy measurement chains (each %d trajectories) for different value pairs beta,L'%(int(num_traj*(1-burnin_frac))))

    e_avg, e_err = np.empty((2, len(betas)))

    t1 = time.time()
    prev_ell, prev_eps = 2, 1/2  # calibration guesses, suitable for small beta.
    for i, beta in enumerate(betas):
        L = int(Ls[i])
        L_str = str(L)
        beta_str = str(np.round(beta, 4)).replace('.', '_')

        # calibrate ell, eps
        model_paras = {'D':D, 'L':L, 'a':1, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=accel)
        prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps'] # update calibration guesses for next largest beta
        model = SU2xSU2(**paras_calibrated) # model for production run

        # set simulation and data storage parameters. Execute production simulation
        file_path = chaindata_pathbase + 'beta%sL%s.npy'%(beta_str, L_str)
        sim_paras = {'M':num_traj, 'burnin_frac':burnin_frac, 'accel':accel, 'measurements':[model.internal_energy_density], 'chain_paths':[file_path]}
        model.run_HMC(**sim_paras) 

        # get ensemble average
        data = np.load(file_path)
        e_avg[i], e_err[i] = get_avg_error(data)

        # write ensemble average to disk
        # first check if path contains any directories and create those if they don't exist already
        dir_path = os.path.dirname(simdata_path)
        if dir_path != '':
            os.makedirs(dir_path, exist_ok=True)
        des_str = '%d measurements on L=%d, a=%d lattice at different beta: beta, avg internal energy and its error.'%(model.M, model.L, model.a)
        np.savetxt(simdata_path, np.row_stack([betas, e_avg, e_err]), header=des_str)
        print('-'*32)
        print('Completed %d / %d: beta=%.3f, L=%d'%(i+1, len(betas), beta, L))
        print('-'*32)
           
    t2 = time.time()
    print('Total simulation time: %s'%(str(timedelta(seconds=t2-t1))))

    # make plot
    plotting.internal_energy_density_plot(D, simdata_path, plot_path, show_plot=False)
    return 


def mass_lambda(betas, D, Ls, num_traj, burnin_frac, accel=True,
                corlengthdata_path='data/corlength.txt', corfuncs_chain_dir='data/corfuncs/rawchains/', corfuncs_dir='data/corfuncs/',
                corfuncs_plot_dir='plots/corfuncs/', plot_path='plots/asymptotic_scaling.pdf'):
    '''
    Computes the mass over lambda parameter ratio for for the passed value pairs of ``betas``, ``Ls`` (lattice size along one dimension) on a D-dimensional lattice.
    The lattice size must be chosen sufficiently large (meaning a multiple of the correlation length) to avoid finite size effects.
    In general, the required lattice size increases quickly with beta.

    Computing the mass over lambda ratio requires the correlation length for each beta,L value pair and will be stored in the text file at ``corlengthdata_path``.
    For each of the considered beta,L value pairs, the raw measurement chain for the correlation function is saved in the directory ``corfuncs_chain_dir``,
    while the the processed correlation function data (normalized as well as ensemble averaged and averaged across the symmetry axis at L/2) 
    is stored at ``corfuncs_dir``. Plots of the correlation functions, with the fitted analytical expectation are stored at ``corfuncs_plot_dir``.
    In all cases the file names are given by considered values of beta,L.

    A plot of mass over lambda ratio is produced and stored at ``plot_path``, allowing to assess the convergence of the simulation data to the continuum mass 
    gap prediction as beta gets large. The beta function is used at 3 loop accuracy and the integral occurring in the definition of the renormalization scale Lambda
    is evaluated numerically.

    Parameters
    ----------
    betas: (n,) array
        values of the model parameter beta for which simulations are performed in ascending order 
    D: int
        dimension of the lattice
    Ls: (n,) array or list
        size of the lattice along one dimension for each simulation at different beta. Must be even integers. 
    num_traj: int
        number of trajectories in each simulation
    burnin_frac: float
        fraction of trajectories discarded as burn in
    accel: bool (optional)
        using Fourier Acceleration by default
    couplingdata_path: str (optional)
        path of .txt file (relative to the current working directory and with file extension) to store the correlation length, its error and the associated chi-squared
        value for all betas,Ls.
    chaindata_pathbase: str (optional)
        path of the directory where the correlation function measurement chains for all value pairs of beta,L will be saved
    corfuncs_dir: str (optional)
        path of the directory where the processed correlation function data for all value pairs of beta,L will be saved
    corfuncs_plot_dir: str (optional)
        path of the directory where the plots of the processed correlation function for all value pairs of beta,L will be saved
    plot_path: str (optional)
        path (with file extension) of the plot showing the numerical mass over lambda ratio against beta.
    '''
    # set up storage path and add readme file explaining the contents of the directory
    # raw correlation function chains
    dir_path = os.path.dirname(corfuncs_chain_dir)
    os.makedirs(dir_path, exist_ok=True)
    with open(corfuncs_chain_dir + 'readme.txt', 'w') as text_file:
        text_file.write('The correlation function measurement chains (each %d trajectories) for different value pairs beta,L'%(int(num_traj*(1-burnin_frac))))

    # processed correlation functions
    dir_path = os.path.dirname(corfuncs_dir)
    os.makedirs(dir_path, exist_ok=True)
    with open(corfuncs_dir + 'readme.txt', 'w') as text_file:
        text_file.write('The correlation function (each %d trajectories) for different value pairs beta,L'%(int(num_traj*(1-burnin_frac))))

    
    xi, xi_err, reduced_chi2 = np.zeros((3,betas.shape[0]))
    prev_ell, prev_eps = 4, 1/4 
    for i,beta in enumerate(betas):
        L = int(Ls[i])
        L_str = str(L)
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        file = 'beta%sL%s.npy'%(beta_str, L_str) # file name and extension for data and plots

        # calibrate ell, eps
        model_paras = {'D':D, 'L':L, 'a':1, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=accel)
        prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps'] # update calibration guesses for next largest beta
        model = SU2xSU2(**paras_calibrated) # model for production run

        # set simulation and data storage parameters. Execute production simulation
        corfunc_chain_path = corfuncs_chain_dir + file
        sim_paras = {'M':num_traj, 'burnin_frac':burnin_frac, 'accel':True, 'measurements':[model.ww_correlation_func], 'chain_paths':[corfunc_chain_path]}
        model.run_HMC(**sim_paras) 
        
        # get ensemble average correlation function. Further processing occurs in get_corlength.
        # Store it and find correlation length
        data = np.load(corfunc_chain_path)
        cor, cor_err = get_avg_error(data) # ensemble avg and err
        corfunc_path = corfuncs_dir + file
        xi[i], xi_err[i], reduced_chi2[i] = get_corlength(cor, cor_err, corfunc_path)
        # plot correlation function
        corfunc_plot_path = (corfuncs_plot_dir + file)[:-4] + '.pdf'
        plotting.correlation_func_plot(corfunc_path, corfunc_plot_path, show_plot=False)

        des_str = 'correlation lengths inferred from %d measurements of the correlation function for different L and beta pairs: D, L, beta, xi, xi_err, chi-square per degree of freedom.'%model.M
        np.savetxt(corlengthdata_path, np.row_stack((np.repeat(D,len(Ls)), Ls, betas, xi, xi_err, reduced_chi2)), header=des_str)
        print('-'*32)
        print('Completed %d / %d: beta=%.3f, L=%d'%(i+1, len(betas), beta, L))
        print('-'*32)

    # make mass over lambda plot
    plotting.mass_lambda_plot(corlengthdata_path, plot_path)
    return


def critical_slowingdown(num_traj, burnin_frac, corlengthdata_path='data/corlength.txt', chi_chain_dir='data/slowdown/rawchains/', 
                         chi_data_dir='data/slowdown/', slowdownplot_path='plots/slowdown.pdf', costfuncplot_path='plots/costfunc.pdf',
                         xticks=[2,5,10,20,50,100], yticks=[1,5,10,25]):
    '''
    Computes the integrated autocorrelation time (IAT) of the susceptibility using beta, L form previous simulations where the correlation length was measured and stored.
    For example, the data produced in :py:func:`SU2xSU2.analysis.mass_lambda` and stored at its argument ``corlengthdata_path`` can be used. 
    Its is assumed that the data has a row wise structure of Ls, betas, correlation length.
    The computation is done for accelerated and unaccelerated dynamics, with the measurement chains being stored in directories ``accel/``, ``unaccel/`` at ``chi_chain_dir``.
    Files are labeled by the used value of beta and L. The ensemble average results are stored as text files at ``chi_data_dir`` while the 
    associated plot of IAT vs correlation length is stored at ``slowdownplot_path``. Power laws are fitted for either acceleration choice, 
    allowing to quantify the degree of critical slowing down through the fitted value of the critical exponent 'z'. 
    A further plot, showing the simulation cost of either acceleration choice, is stored at ``costfuncplot_path``. 
     
    Parameters
    ----------
    num_traj: int
        number of trajectories in each simulation
    burnin_frac: float
        fraction of trajectories discarded as burn in
    corlengthdata_path: str (optional)
        path to the Ds, Ls, betas, correlation length data (must be .txt). The default causes to use the result from :py:func:`SU2xSU2.analysis.mass_lambda`
        when its argument with the same name is also left as default.
    chi_chain_dir: str (optional)
        path of directory where two directories (named accel and unaccel) will be created to store the measurement chains of the susceptibility.
    chi_data_dir: str (optional)
        path of directory where two directories (named accel and unaccel) will be created to store L, beta, the IAT, its error, 
        the ensemble averaged susceptibility, its error, the simulation time, and the acceptance rate (row wise).
    slowdownplot_path: str (optional)
        path to plot showing the power law scaling of the susceptibility IAT with the correlation length 
    costfuncplot_path: str (optional)
        path to save the simulation cost function plot at
    xticks: list (optional)
        list of ints or floats specifying the tick labels for the correlation length in the two plots
    yticks: list (optional)
        list of ints or floats specifying the tick labels for the ratio unaccel cost function / accel cost function in the cost function plot
    '''
    data = np.loadtxt(corlengthdata_path)
    Ds, Ls, betas, xis = data[:4]
  
    accel_bool = [False, True]
    n = len(betas)
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc = np.zeros((2,n)), np.zeros((2,n))

    chain_dir = [chi_chain_dir+'unaccel/', chi_chain_dir+'accel/']
    file_path = [chi_data_dir+'unaccel.txt', chi_data_dir+'accel.txt'] # location for final results depending on use of acceleration

    # create directories in file_path if necessary
    for path in file_path:
        dir_path = os.path.dirname(path)
        if dir_path != '':
            os.makedirs(dir_path, exist_ok=True)

    description = 'dynamics with %d total measurements of susceptibility \nL, beta, IAT and error, chi and error, simulation time [sec], acceptance rate'%num_traj
    des_str = ['Unaccelerated '+description, 'Accelerated '+description]

    prev_ell, prev_eps = [4,4], [1/4, 1/4] 
    for i,beta in enumerate(betas):
        D = int(Ds[i])
        L = int(Ls[i])
        L_str = str(L)
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        file = 'beta%sL%s.npy'%(beta_str, L_str) # file name and extension for data 
        for k, accel in enumerate(accel_bool):
            # calibrate ell, eps
            model_paras = {'D':D, 'L':L, 'a':1, 'ell':prev_ell[k], 'eps':prev_eps[k], 'beta':beta, 'mass':1/xis[i]}
            paras_calibrated = calibrate(model_paras, accel=accel)
            prev_ell[k], prev_eps[k] = paras_calibrated['ell'], paras_calibrated['eps']
            model = SU2xSU2(**paras_calibrated)
        
            # production simulation and saving raw chain of susceptibility measurements
            chain_path = chain_dir[k] + file 
            sim_paras = {'M':num_traj, 'burnin_frac':burnin_frac, 'accel':accel, 'measurements':[model.susceptibility], 'chain_paths':[chain_path]}
            model.run_HMC(**sim_paras) 

            # get ensemble average and IAT of susceptibility
            data = np.load(chain_path) 
            chis[k,i], chis_err[k,i], IATs[k,i], IATs_err[k,i] = get_avg_error(data, get_IAT=True)
            times[k,i], acc[k,i] = model.time, model.acc_rate

            np.savetxt(file_path[k], np.row_stack((Ls, betas, IATs[k], IATs_err[k], chis[k], chis_err[k], times[k], acc[k])), header=des_str[k])
        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)


    ### make plots ### accel_data_path, unaccel_data_path, corlengthdata_path, xticks, plot_path, show_plot=True
    plotting.critical_slowing_plot(file_path[0], file_path[1], corlengthdata_path, xticks, slowdownplot_path)
    plotting.cost_function_ratio_plot(file_path[0], file_path[1], corlengthdata_path, xticks, yticks, costfuncplot_path)
    return


def acceleration_mass_search(num_traj, burnin_frac, beta, D, L, corlength, masses,
                            chain_dir='data/acceleration_mass/', data_path='data/accelertion_mass.txt', plot_path='plots/acceleration_mass.pdf'):
    '''
    Performs a grid search of the mass acceleration parameter for the passed value pair of ``beta``, ``L``.
    To assess the quality of the canonical choice of M=1/correlation length against other values of the parameter, a plot is produced
    comparing the simulation cost agaist values of the acceleration mass. The plot is stored at ``plot_path`` and the associated 
    data is stored at ``data_path``. Computing the cost function requires measurming the susceptibility integrated autocorrelation time.
    The susceptibility chain is stored in the directory ``chain_dir`` with the file being labeled by the values of ``beta``, ``L`` and the considered value of the mass.

    Parameters
    ----------
    num_traj: int
        number of trajectories in each simulation using a different acceleration mass
    burnin_frac: float
        fraction of trajectories discarded as burn in
    beta: float
        model parameter beta for which the simulations are performed
    D: int
        dimension of the lattice
    L: int
        lattice size along one direction. Must be even.
    corlength: float
        correlation length infered for simulations at beta
    masses: array
        values of the acceleration parameter to include in the grid search
    chain_dir: str (optional)
        path to directory containing susceptibility measurement chain.
    data_path: str (optional)
        path at which the collected data will be stored: masses, cost function and its error, simulation time, acceptance rate, susceptbility IAT and its error.
        Must be a text file.
    plot_path: str (optional)
        path at which the plot of the cost fucntion vs the acceleration mass is stored
    '''
    masses = np.append(masses, 1/corlength)
    n = masses.size
    times, acc_rates, chi_IATs, chi_IATs_err = np.zeros((4,n))
    cost_func, cost_func_err = np.zeros((2,n))

    # arbitary initial guesses for calibration
    prev_ell, prev_eps = 10, 1/10
    for i,mass in enumerate(masses):
        L_str = str(L)
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        mass_str = str(np.round(mass, 4)).replace('.', '_')
        file = 'beta%sL%sm%s.npy'%(beta_str, L_str, mass_str) # file name and extension for measurement chain
        # calibrate ell, eps
        model_paras = {'D':D, 'L':L, 'a':1, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta, 'mass':mass}
        paras_calibrated = calibrate(model_paras, accel=True)
        prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps'] # update calibration guesses
        model = SU2xSU2(**paras_calibrated) # model for production run

        # production simulation and saving raw chain of susceptibility measurements
        chain_path = chain_dir + file 
        sim_paras = {'M':num_traj, 'burnin_frac':burnin_frac, 'accel':True, 'measurements':[model.susceptibility], 'chain_paths':[chain_path]}
        model.run_HMC(**sim_paras) 

        # compute cost function 
        data = np.load(chain_path) 
        _, _, chi_IATs[i], chi_IATs_err[i] = get_avg_error(data, get_IAT=True)
        times[i], acc_rates[i] = model.time, model.acc_rate
        cost_func[i] = times[i] / acc_rates[i] * np.sqrt(chi_IATs[i])
        cost_func_err[i] = cost_func[i] * 1/2 * chi_IATs_err[i]/chi_IATs[i]
    
        header_str = 'acceleration masses, non-normalised cost function and its error, simulation time [sec], acceptance rate, susceptibility IAT and its error'
        # create directories in data_path if necessary
        dir_path = os.path.dirname(data_path)
        if dir_path != '':
            os.makedirs(dir_path, exist_ok=True)
        np.savetxt(data_path, np.row_stack((masses, cost_func, cost_func_err, times, acc_rates, chi_IATs, chi_IATs_err)), header=header_str)
        print('Completed %d/%d'%(i+1,n))

    idx = np.argmin(cost_func)
    print('Value of mass parameter yielding most efficient acceleration: M = %.5f'%masses[idx])

    # make plot where cost function is normalized to M=1/corlentgh
    cost_func /= cost_func[-1]
    cost_func_err /= cost_func[-1] 

    fig = plt.figure()
    plt.errorbar(masses[:-1], cost_func[:-1], yerr=cost_func_err[:-1], fmt='.')
    plt.errorbar(masses[-1], cost_func[-1], yerr=cost_func_err[-1], c='r', fmt='x', markersize=4, label=r'$M=1/\xi$')
    plt.xlabel(r'acceleration mass $M$')
    plt.ylabel(r'cost function normalised to $M=1/\xi$')
    plt.legend()
    # create directories in plot_path if necessary
    dir_path = os.path.dirname(plot_path)
    if dir_path != '':
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(plot_path)
    plt.close()