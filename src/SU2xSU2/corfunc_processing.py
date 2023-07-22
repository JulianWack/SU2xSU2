# routines to process stored correlation function data and make related plots
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import get_avg_error

plt.style.use('science')
plt.rcParams.update({'font.size': 30})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])
mpl.rcParams['xtick.major.size'], mpl.rcParams['ytick.major.size'] = 10, 10
mpl.rcParams['xtick.minor.size'], mpl.rcParams['ytick.minor.size'] = 5, 5



def process_rawchain():
    '''
    Processes raw correlation function data, by finding the ensemble average, normalizing data and averaging it about its symmetry axis at L/2, increasing
    the statistics by a factor 2. The resulting data is stored at a hard coded path.
    Recommended to be used on data from partially complete simulations (identically functionality is contained in SU2xSU2.corlength). 
    '''
    beta_str = '1_4'
    corfunc_chain = np.load('data/corfuncs/rawchains/%s.npy'%beta_str)
    ww_cor, ww_cor_err = get_avg_error(corfunc_chain)
    print('data shape: ', corfunc_chain.shape)

    # normalize and use periodic bcs to get correlation for wall separation of L to equal that of separation 0
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    ww_cor, ww_cor_err = np.concatenate((ww_cor, [ww_cor[0]])), np.concatenate((ww_cor_err, [ww_cor_err[0]]))

    # use symmetry about L/2 due to periodic bcs and mirror the data to reduce errors (effectively increasing number of data points by factor of 2)
    N_2 = int(ww_cor.shape[0]/2) 
    ds = np.arange(N_2+1) # wall separations covering half the lattice length
    cor = 1/2 * (ww_cor[:N_2+1] + ww_cor[N_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:N_2+1]**2 + ww_cor_err[N_2::-1]**2) / np.sqrt(2)

    np.save('data/corfuncs/beta_%s.npy'%beta_str, np.row_stack([ds, cor, cor_err]))

# process_rawchain()


def effective_mass(beta):
    '''
    Produces an effective mass plot form the correlation function data collected during the run with the passed value of beta.
    The normalized and averaged (ensemble and symmetry axis) correlation function data is used to produce the plot, either based on the 
    assumption that the correlation function follows the shape of a cosh (analytically expected due to periodic boundary conditions) or of a pure exponential decay.

    For small physical separations the two methods should agree, allowing to gauge if significant finite size effects are present.
    The cosh assumption will generally produce a noisier plot as each data point considers 3 values of the correlation function while 
    in the decay assumption only two are used.

    Parameters
    ----------
    beta: float
        value of the model parameter beta used in the considered simulation run
    '''
    def cosh_corfunc(cor, cor_err):
        '''
        Finds the effective mass and its error based on a cosh correlation function.
        A lattice of even size is assumed.

        Parameters
        ----------
        cor: (L/2)
            value of wall to wall correlation function on the first half of the lattice
        cor_err: (L/2)
            error of correlation function on the first half of the lattice

        Returns
        -------
        m_eff: (L/2,)
            effective mass
        m_eff_err: (L/2,)
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
        m_eff_err = 1/np.sqrt(x**2-1) * delta_x

        return m_eff, m_eff_err

    def exp_corfunc(cor, cor_err):
        '''
        Finds the effective mass and its error based on an exp correlation function.
        A lattice of even size is assumed.

        Parameters
        ----------
        cor: (L/2,)
            value of  wall to wall correlation function on the first half of the lattice
        cor_err: (L/2,)
            error of correlation function on the first half of the lattice

        Returns
        m_eff: (L/2,)
            effective mass
        m_eff_err: (L/2,)
            error of the effective mass
        '''
        cor_1 = np.roll(cor, -1) # shift to d+1
        m_eff = - np.log(cor_1 / cor)
        m_eff_err = np.roll(cor_err, -1)/cor_1 - cor_err/cor 
        # m_eff_err = np.sqrt( (np.roll(ww_cor_err_mirrored, -1)/cor_1)**2 - (ww_cor_err_mirrored/cor)**2 )

        return m_eff, m_eff_err


    beta_str = str(np.round(beta, 4)).replace('.', '_')
    ds_2, cor, cor_err = np.load('data/corfuncs/beta_%s.npy'%beta_str)
    
    m_eff_cosh, m_eff_err_cosh = cosh_corfunc(cor, cor_err)
    m_eff_exp, m_eff_err_exp = exp_corfunc(cor, cor_err)

    fig = plt.figure(figsize=(8,6))

    cut = 200 # adjust manually
    plt.errorbar(ds_2[:cut], m_eff_cosh[:cut], yerr=m_eff_err_cosh[:cut], fmt='.', capsize=2, label='$\cosh$', c='red')
    plt.errorbar(ds_2[:cut]-0.2, m_eff_exp[:cut], yerr=m_eff_err_exp[:cut], fmt='.', capsize=2, label='$\exp$', c='b') # slightly shift data points to avoid overlapping
    # plt.ylim(bottom=0.1, top=0.183)
    # plt.ylim(bottom=0.3)
    # plt.ylim(top=0.625)
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel('effective mass $m_{eff}$')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.legend(prop={'size': 12}, frameon=True)
    plt.show()
    # fig.savefig('plots/corfuncs/effective_mass/%s.pdf'%beta_str)

    return

# values of beta
# [0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4]
# effective_mass(1.4)