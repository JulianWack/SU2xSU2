# some additional plotting routines
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# plt.style.use('scientific.mplstyle')
# plt.rcParams.update({'text.usetex': True}) # uncomment when latex is installed


def plot_chain(data_chain_path, ylabel, start_idx=0):
    '''
    Plots raw measurement chain against computer time to gauge when convergence has been achieved.
    Observables of particular interest are the susceptibility (as sufficient burn is needed for the critical slowing down analysis) and 
    the correlation function at a fixed separation (ideally chosen close to the correlation length as this is a slowly converging quantity
    and thus gives a lower bound for burn in).
    If the stored chain has already had some burn in removed, the starting value of the computer time can be adjusted by 'start_idx'. 

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
    Produces a plot (with a logarithmic y axis) of the correlation function data (stored at 'data_path') and the fitted, analytical expectation. 
    The plot is saved at 'plot_path' with the fitting range and other plot parameters can be adjusted through the arguments.

    Allows manual adjustment of fitting the correlation length to the processed correlation function data (normalized and mirror averaged).
    A plot with the new fitting is produced and the inferred correlation length, its error and the associated chi2 are printed.
    These can then be manually added to the data/corlen_data.txt file.

    Parameters
    ----------
    data_path: str
        file path to the correlation function data that will be plotted. The file is assumed to be .npy and 
        contain the rows separation [0,L/2], correlation function, correlation function error respectively 
    plot_path: str 
        file path (including file extension) to store the plot at
    fit_upper: int (optional)
        largest separation (in units of the lattice spacing) to be included in the fit.
        If left as 'None', only all non-zero values of the correlation function will be included in the fit
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


def effective_mass_plot(data_path, xright=None, ytop=None, ybottom=None, show_plot=True, plot_path='plots/effective_mass.pdf'):
    '''
    Produces an effective mass plot using the processed correlation function data stored at 'data_path'.
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


