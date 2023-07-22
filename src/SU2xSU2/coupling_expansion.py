import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import time
from datetime import timedelta

from SU2xSU2 import SU2xSU2, get_avg_error
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])



def internal_energy_coupling_exp(betas):
    '''Computes and stores the internal energy per site for the passed values of beta. 
    A plot is produced comparing the simulation result with the weak and strong coupling expansions (w.c. and s.c.).
    
    Parameters
    ----------
    betas: (n,) array
        value of beta to run simulations for
    '''
    # Ns = [40, 40, 64, 64, 64, 96, 96, 160, 160, 224, 400, 512, 700]
    # betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4])
    Ns = np.full(betas.shape, 16, dtype=int)

    e_avg, e_err = np.empty((2, len(betas)))

    t1 = time.time()
    prev_ell, prev_eps = 2, 1/2  # calibration guesses, suitable for small beta. Get updated to previous simulation during loop over betas
    for i, beta in enumerate(betas):
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        model_paras = {'N':Ns[i], 'a':1, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=True)
        prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps']

        model = SU2xSU2(**paras_calibrated)
        # file_path = 'data/energy_density_E_scheme/beta_'+beta_str
        file_path = 'data/energy_density/beta_'+beta_str
        sim_paras = {'M':5000, 'thin_freq':1, 'burnin_frac':0.1, 'accel':True, 'measurements':[model.internal_energy_density], 'chain_paths':[file_path]}
        model.run_HMC(**sim_paras) 

        # get ensemble average
        data = np.load(file_path+'.npy')
        e_avg[i], e_err[i] = get_avg_error(data)

        des_str = '%d measurements on N=%d, a=%d lattice at different beta: beta, avg internal energy and its error.'%(model.M, model.N, model.a)
        np.savetxt('data/coupling_expansion.txt', np.row_stack([betas, e_avg, e_err]), header=des_str)
        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)
           
    t2 = time.time()
    print('Total time: %s'%(str(timedelta(seconds=t2-t1))))


    # make strong and weak coupling expansion plot
    b_s = np.linspace(0,1)
    strong = 1/2*b_s + 1/6*b_s**3 + 1/6*b_s**5

    Q1 = 0.0958876
    Q2 = -0.0670
    b_w = np.linspace(0.6, 4)
    weak = 1 - 3/(8*b_w) * (1 + 1/(16*b_w) + (1/64 + 3/16*Q1 + 1/8*Q2)/b_w**2)

    fig = plt.figure(figsize=(16,6))

    plt.errorbar(betas, e_avg, yerr=e_err, fmt='x', capsize=2, label='FA HMC')
    plt.plot(b_s, strong, c='b', label='s.c.')
    plt.plot(b_w, weak, c='r', label='w.c.')
    plt.xlabel(r'$\beta$')
    plt.ylabel('internal energy density $e$')
    plt.legend(prop={'size': 12}, frameon=True)

    plt.show()
    # fig.savefig('plots/coupling_expansion.pdf')
    return 
    

# betas = np.linspace(0.01, 4, 20)
betas = np.array([0.01, 0.22, 0.43, 0.52, 0.61, 0.7, 0.79, 0.88, 0.97, 1.06, 1.27, 1.48, 1.69, 1.9,  2.11, 2.32, 2.53, 2.74, 2.95, 3.16, 3.37, 3.58, 3.79, 4])

# betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4])
internal_energy_coupling_exp(betas)