import numpy as np
from SU2xSU2 import SU2xSU2


def calibrate(model_paras, accel=False, sim_paras=None):
    '''For a model, specified by the dictionary model_paras, this function calibrates the values of ell and eps to produce an acceptance rate in the desireable range between 60 and 75%.
    When acceptance rate is outside this range, the number of steps is adjusted according to the difference to the ideal acceptance rate of 65%. The step size if fixed by requiring
    trajectories to be of unit length. To avoid getting caught in a loop, the calibration is limited to 10 iterations.
    When sim_paras is not passed, a default calibration is performed: 500 trajectories (no thinning and 50% burn in) are simulated and no measurements are taken.
    Passing a dictionary for sim_paras overwrites this.

    Recommended use: For the model of interest, specify the acceleration boolean and leave the simulation parameters as a default. Use the returned calibrated parameters to perform a
    production run i.e. call run_HMC(**sim_paras) where sim_paras specifies the chain and measurements you want to take.
    One can also perform the production run through this function by passing sim_paras.
    
    model_paras: dict
        {N, a, ell, eps, beta} with ell, eps as guesses to start the calibration. Their product must be 1
    accel: bool
        use acceleration or not
    sim_paras: dict
        arguments of SU2xSU2.run_HMC, namely
        {M, thin_freq, burnin_frac, accel=True, measurements=[], chain_paths=[], saving_bool=True, partial_save=5000, starting_config=None, RGN_state=None, renorm_freq=10000}
        
    Returns
    model_paras: dict
        calibrated model parameters
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