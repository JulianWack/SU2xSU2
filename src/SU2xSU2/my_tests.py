import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import get_avg_error

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def mass_eff_finite_spacing():
    '''Illustrates a finite lattice spacing effect in the effective mass plot'''
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
        m_eff_err = 1/np.sqrt(x**2-1) * delta_x

        return m_eff, m_eff_err
    
    ds_2, cor, cor_err = np.load('data/mixed_data/beta_1_1333a_0_4.npy')
    m_eff_small, m_eff_err_small = cosh_corfunc(cor, cor_err)

    ds_2, cor, cor_err = np.load('data/mixed_data/beta_1_1333a_1.npy')
    m_eff, m_eff_err = cosh_corfunc(cor, cor_err)

    fig = plt.figure(figsize=(16,9))

    cut = 22 # adjust manually
    plt.errorbar(ds_2[:cut], m_eff[:cut], yerr=m_eff_err[:cut], fmt='.', capsize=2, label='$a=1$', c='b')
    plt.errorbar(ds_2[:cut], m_eff_small[:cut], yerr=m_eff_err_small[:cut], fmt='.', capsize=2, label='$a=0.4$', c='red')
    # plt.ylim(bottom=0.054, top=0.085)
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel('effective mass $m_{eff}$')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.legend(prop={'size': 18}, frameon=True)
    plt.show()
    # fig.savefig('plots/corfuncs/effective_mass/%s.pdf'%beta_str)

    return

# mass_eff_finite_spacing()
