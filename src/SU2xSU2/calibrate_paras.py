import numpy as np
from .SU2xSU2 import SU2xSU2


def calibrate(model_paras, accel=False, sim_paras=None):
    '''
    For a model, specified by the dictionary model_paras, this function calibrates the number of leapfrog integration steps and their size 
    under the constraint that the trajectory length is 1 and that the acceptance rate is with in desireable range between 60 and 75%.
    
    The calibration is done by performing short simulation runs (500 trajectories with 50% burn in unless overridden by passing ``sim_paras``), extracting the acceptance rate
    and, if the acceptance rate is outside this range, adjusting the number of steps according to the difference to the ideal acceptance rate of 65% for the next run. The 
    number of steps is inferred from constraining the trajectory length to unity.
    It is not guaranteed that the calibration is successful for all possible model specification given the fixed trajectory length. Hence the calibration is limited to 10 
    iterations. An indicator that longer trajectories are required is when the calibration algorithm tries to reduce the number of steps below one.

    Recommended to use the returned calibrated ``model_paras`` to define the model for the production run. 
            
    Parameters
    ----------
    model_paras: dict
        {L, a, ell, eps, beta} denoting lattice size, lattice spacing, number of integration steps, integration step size and the SU(2)xSU(2) model parameter beta respectively
        The values of ell, eps are used as guesses to start the calibration and their product must be 1.
    accel: bool, optional
        use Fourier Acceleration or not
    sim_paras: dict, optional
        {M, thin_freq, burnin_frac, accel=True, measurements=[], chain_paths=[], saving_bool=True, partial_save=5000, starting_config=None, RGN_state=None, renorm_freq=10000}
        Specifying the simulation parameters for the calibration run by calling *SU2xSU2.run_HMC*. 
        Consider the associated docstring for definitions of these parameters.

    Returns
    -------
    model_paras: dict
        calibrated model parameters and of the same form as model_paras
    '''
    # defining bounds for desireable acceptance rate
    lower_acc, upper_acc = 0.6, 0.75

    if sim_paras is None:
        # default for fast calibration
        sim_paras = {'M':500, 'thin_freq':1, 'burnin_frac':0.5, 'accel':accel, 'saving_bool':False}
    
    good_acc_rate = False
    count = 0 
    stop_flag = False
    while good_acc_rate == False:
        model = SU2xSU2(**model_paras)
        model.run_HMC(**sim_paras)  
        acc_rate = model.acc_rate
        d_acc_rate = 0.65 - acc_rate
        if count >= 10:
            good_acc_rate = True
        if acc_rate < lower_acc or acc_rate > upper_acc:
            new_ell = int(np.rint(model_paras['ell']*(1 + d_acc_rate)))
            # due to rounding it can happen that ell is not updated. To avoid getting stuck in a loop, enforce minimal update of +/- 1
            if new_ell == model_paras['ell']:
                if d_acc_rate > 0:
                    new_ell += 1
                else:
                    new_ell -= 1
                    if new_ell == 0:
                        # stop calibration when step size has to be reduce below 1.
                        stop_flag = True
                        break  
            model_paras['ell'] = new_ell
            model_paras['eps'] = 1/model_paras['ell']
            count +=1
        else:
            good_acc_rate = True

    stats = 'acc= %.2f%%, ell=%d, eps=%.3f'%(acc_rate*100, model_paras['ell'], model_paras['eps'])
    if count < 10 and not stop_flag:
        print('Successful calibration: '+stats)
    else:
        print('Unsuccessful calibration: '+stats)

    return model_paras