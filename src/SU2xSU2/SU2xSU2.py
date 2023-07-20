import numpy as np
from pickle import dump, load 
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler
import time
from datetime import timedelta
from scipy.optimize import curve_fit

import SU2_mat_routines as SU2
import correlations


plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


# simulation class
class SU2xSU2():

    def __init__(self, N, a, ell, eps, beta, mass=0.1): 
        '''
        N: int
            Number of lattice sites along one dimension. Must be even for implementation of Fourier acceleration to work properly 
        a: float
            Lattice spacing
        ell: int
            Number of steps to integrate Hamilton's equation, each of size eps
        eps: float
            Step size for integrating Hamilton's equations
        beta: float
            nearest neighbor coupling parameter
        mass: float
            mass parameter used in Fourier acceleration. Given default value was fund to yield most effective acceleration
        '''

        # lattice parameters
        self.N, self.a = int(N), a
        # leapfrog parameters
        self.ell, self.eps = int(ell), eps
        # model parameters
        self.beta = beta
        # acceleration parameters
        self.mass = mass

        # find mask to index phi giving the parameters of the right, left, top, and bottom nearest neighbor
        self.NN_mask = self.make_NN_mask() 


    def make_NN_mask(self):
        '''Makes mask to apply to phi or pi which then gives the matrix parameter values of the nearest neighbors (NN) for each lattice site.
        Hence phi[self.NN_mask] is of shape (N,N,#neighbors,#parameters) i.e (N,N,4,4).
        
        Returns:
        NN_mask: tuple
            tuple of two (N,N,4,1) arrays, each giving the row and column coordinate for all nearest neighbors
        '''
     
        # make a (N,N,2) array storing the row and col indices of each lattice sites
        grid = np.indices((self.N,self.N)) 
        lattice_coords = grid.transpose(1,2,0)

        # shift lattice coordinates by 1 such that the coordinates at (i,j) are those of the right, left, top, and bottom neighbor of lattice site (i,j)
        # rolling axis=1 by -1 means all columns are moved one step to the left with periodic bcs. Hence value of resulting array at (i,j) is (i,j+1), i.e the coordinates of the right neighbor.
        # all of shape (N,N,2)
        right_n = np.roll(lattice_coords, -1, axis=1)
        left_n = np.roll(lattice_coords, 1, axis=1)
        top_n = np.roll(lattice_coords, 1, axis=0)
        bottom_n = np.roll(lattice_coords, -1, axis=0)

        # for each lattice site, for each neighbor, store row and column coordinates
        # order of neighbors: right, left, top, bottom
        NN = np.empty((self.N,self.N,4,2), dtype=int)
        NN[:,:,0,:] = right_n # row and col indices of right neighbors
        NN[:,:,1,:] = left_n
        NN[:,:,2,:] = top_n
        NN[:,:,3,:] = bottom_n

        # make mask to index phi
        # separate the row and column neighbor coordinates for each lattice site: (N,N,4,1)
        NN_rows = NN[:,:,:,0]
        NN_cols = NN[:,:,:,1]
        NN_mask = (NN_rows, NN_cols)

        return NN_mask 

        
    def action(self, phi):
        '''
        Computes the action for lattice configuration phi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site

        Returns
        S: float
            the action
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask] # (N,N,4,4): containing the 4 paras of each of the 4 NN

        # sum over lattice unit vectors: to the right and up. Hence only need right and top NN, stored at position 0,3 respectively
        G = np.zeros((self.N,self.N))
        for i in [0,3]:
            A = SU2.dot(phi_hc, phi_NN[:,:,i,:])
            G += SU2.tr(A + SU2.hc(A)) # when getting UFuncTypeError, check that dtype of G and SU2.tr is the same (float64 by default)

        # sum over lattice sites    
        S = -1/2 * self.beta * np.sum(G)

        return S


    def Ham(self, phi, pi):
        '''
        Computes the Hamiltonian for a lattice configuration phi, pi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site
        pi: (N,N,3) array
            parameter values conjugate momenta at each lattice site
            
        Returns
        H: float
            the Hamiltonian as the sum of the action and a kinetic term, quadratic in pi
        '''
        T = 1/2 * np.sum(pi**2) # equivalent to first summing the square of the parameters at each site and then sum over all sites
        S = self.action(phi)
        H = T + S

        return H 


    def Ham_FA(self, phi, pi):
        '''Analogous to function self.Ham but computes the modified hamiltonian used to accelerate the dynamics.
        '''
        pi_F_mag = np.sum( np.abs(np.fft.fft2(pi, axes=(0,1)))**2, axis=-1 ) # (N,N) find magnitude of FT of each component of momentum in Fourier space. Then sum over all 3 components
        T = 1/(2*self.N**2) * np.sum(pi_F_mag*self.A) # sum over momentum Fourier lattice
        S = self.action(phi)
        H = T + S 

        return H


    def prod_A_pi(self, pi_F):
        '''Computes the element wise product of the inverse kernel and the momenta in Fourier space.
        In the literature often written as the element wise product of A and pi.

        pi_F: (N,N,3) array
            parameter vector of momenta in Fourier space

        Returns
            parameter vector of momenta in Fourier space, each site being weighted by the inverse Fourier space kernel
        '''
        return np.multiply(self.A.reshape((self.N,self.N,1)), pi_F)


    def kernel_inv_F(self):
        '''Finds inverse of the action kernel computed in the Fourier space, usually referred to as 'A'.

        Returns
        A: (N,N) array
            inverse kernel in Fourier space
        '''
        # x = 0.9 # parameter interpolating between accelerated (x=1) and unaccelerated (x=0) case. 
        # Appropriate kernel: A[k,k_] = (1 - x/2 - x/4*(np.cos(np.pi*ks[k]/self.N) + np.cos(np.pi*ks[k_]/self.N)) )**(-1)
        ks = np.arange(0, self.N) # lattice sites in Fourier space along one direction
        A = np.zeros((self.N,self.N)) # inverse kernel computed at every site in Fourier space
        for k in range(self.N):
            for k_ in range(k,self.N):
                A[k,k_] = ( 4*np.sin(np.pi*ks[k]/self.N)**2 + 4*np.sin(np.pi*ks[k_]/self.N)**2 + self.mass**2)**(-1)   
                A[k_,k] = A[k,k_] # exploit symmetry of kernel under exchange of directions 

        return A


    def pi_dot(self, phi):
        '''Time derivative of pi which is given as i times the derivative of the action wrt. phi.
        pi and pi dot are linear combinations of the Pauli matrices and hence described by 3 real parameters alpha
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask]
        # need sum of NN pairs along the two lattice unit vectors i.e. right+left and top+bottom
        alpha = np.zeros((self.N, self.N, 3))
        for pos, neg in zip([0,1], [2,3]):
            # sum is proportional to SU2 matrix, allowing to apply the SU2 product routine once proportionality constant has been identified
            sum_in_SU2, prop_const = SU2.sum(phi_NN[:,:,pos,:], phi_NN[:,:,neg,:]) # both are potentially complex but their product is always real
            V = (prop_const * SU2.dot(sum_in_SU2, phi_hc)).real
            alpha += 2*V[:,:,1:] # 3 parameters describing matrix -i(V - V^dagger) for the currently considered direction in the lattice

        return self.beta * alpha


    def exp_update(self, pi_dot_dt):
        '''The update matrix for the field phi is the exponential of a linear combination of generators i.e. an SU(2) element itself.
        SU(2) is special as this exponential can be evaluated exactly.

        Returns:
        update: (N,N,4) array
            parameter vectors of the matrices to update phi
        '''
        return SU2.alpha_to_a(pi_dot_dt)


    def leapfrog(self, phi_old, pi_old):
        '''
        Returns a new candidate lattice configuration and conjugate momenta by evolving the passed configuration and momenta via Hamilton's equations through the leapfrog scheme.
        phi_old: (N,N,4) array
            last accepted sample of SU(2) matrices (specifically their parameter vectors) at each lattice site
        pi_old: (N,N,3) array
            conjugate momenta (specifically their parameter vectors) corresponding to phi_old
            
        Returns:
        phi_cur: (N,N,4) array
            SU(2) matrix parameter vectors after simulating dynamics
        pi_cur: (N,N,3) array
            momenta parameter vectors after simulating dynamics
        '''
        # half step in pi, full step in phi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_old)
        pi_cur = pi_old + pi_dot_dt_half
        phi_cur = SU2.dot(self.exp_update(pi_cur*self.eps), phi_old)

        # ell-1 alternating full steps
        for n in range(self.ell):
            pi_dot_dt = self.eps * self.pi_dot(phi_cur)
            pi_cur = pi_cur + pi_dot_dt
            phi_cur = SU2.dot(self.exp_update(pi_cur*self.eps), phi_cur)
    
        # half step in pi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_cur)
        pi_cur = pi_cur + pi_dot_dt_half

        return phi_cur, pi_cur


    def leapfrog_FA(self, phi_old, pi_old):
        '''
        Analogous to self.leapfrog but uses the modified EoMs.
        '''
        def pi_FA(pi):
            '''Computes the modified momentum term entering in the exponential update in the accelerated dynamics.
            The modified momentum is given by the ordinary momentum pi, weighted by the inverse kernel which is easiest computed in Fourier space.'''
            pi_F = np.fft.fft2(pi, axes=(0,1))
            return np.real( np.fft.ifft2(self.prod_A_pi(pi_F), axes=(0,1)) )

        # half step in pi, full step in phi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_old)
        pi_cur = pi_old + pi_dot_dt_half
        phi_cur = SU2.dot( self.exp_update(pi_FA(pi_cur)*self.eps), phi_old )

        # ell-1 alternating full steps
        for n in range(self.ell):
            pi_dot_dt = self.eps * self.pi_dot(phi_cur)
            pi_cur = pi_cur + pi_dot_dt
            phi_cur = SU2.dot( self.exp_update(pi_FA(pi_cur)*self.eps), phi_cur )
    
        # half step in pi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_cur)
        pi_cur = pi_cur + pi_dot_dt_half

        return phi_cur, pi_cur


    def pi_samples(self):
        '''Returns real space sample of momenta according to the distribution based on the modified kinetic term in the modified hamiltonian.
        N=even is assumed.
        Process of mapping between the PI and pi_F only depends on the lattice position (axes 0 and 1) but not on the component considered (axis 2). 
        Hence, axis 2 does not need to be dealt with explicitly.

        Returns
        pi: (N,N,3) array
            samples of the auxillary momentum parameter vector in real space
        '''
        # momenta in Fourier space
        pi_F = np.zeros((self.N, self.N, 3), dtype=complex)

        PI_std = np.sqrt(self.N**2 / self.A) 
        STD = np.repeat(PI_std[:,:,None], repeats=3, axis=2) # standard deviation is identical for components at same position
        PI = np.random.normal(loc=0, scale=STD) #  (N,N,3) as returned array matches shape of STD

        # assign special modes for which FT exponential becomes +/-1. To get real pi in real space, the modes must be real themselves.
        N_2 = int(self.N/2)
        # two spacial indices
        pi_F[0,0] = PI[0,0]
        pi_F[0,N_2] = PI[0,N_2]
        pi_F[N_2,0] = PI[N_2,0]
        pi_F[N_2,N_2] = PI[N_2,N_2]

        # one special index
        pi_F[0,1:N_2] = 1/np.sqrt(2) * (PI[0,1:N_2] + 1j * PI[0,N_2+1:][::-1])
        pi_F[0,N_2+1:] = np.conj(pi_F[0,1:N_2][::-1]) # imposing hermitean symmetry

        pi_F[N_2,1:N_2] = 1/np.sqrt(2) * (PI[N_2,1:N_2] + 1j * PI[N_2,N_2+1:][::-1])
        pi_F[N_2,N_2+1:] = np.conj(pi_F[N_2,1:N_2][::-1])

        pi_F[1:N_2,0] = 1/np.sqrt(2) * (PI[1:N_2,0] + 1j * PI[N_2+1:,0][::-1])
        pi_F[N_2+1:,0] = np.conj(pi_F[1:N_2,0][::-1])

        pi_F[1:N_2,N_2] = 1/np.sqrt(2) * (PI[1:N_2,N_2] + 1j * PI[N_2+1:,N_2][::-1])
        pi_F[N_2+1:,N_2] = np.conj(pi_F[1:N_2,N_2][::-1])

        # no special index
        pi_F[1:N_2,1:N_2] = 1/np.sqrt(2) * (PI[1:N_2,1:N_2] + 1j * PI[N_2+1:,N_2+1:][::-1,::-1])
        pi_F[N_2+1:,N_2+1:] = np.conj(pi_F[1:N_2,1:N_2][::-1,::-1]) # imposing hermitean symmetry
   
        pi_F[1:N_2,N_2+1:] = 1/np.sqrt(2) * (PI[1:N_2,N_2+1:] + 1j * PI[N_2+1:,1:N_2][::-1,::-1])
        pi_F[N_2+1:,1:N_2] = np.conj(pi_F[1:N_2,N_2+1:][::-1,::-1])

        # pi is real by construction
        pi = np.real(np.fft.ifft2(pi_F, axes=(0,1)))

        return pi


    def run_HMC(self, M, thin_freq, burnin_frac, accel=True, measurements=[], chain_paths=[], saving_bool=True, partial_save=5000, starting_config=None, RGN_state=None, renorm_freq=10000):
        '''Perform the HMC algorithm to generate lattice configurations using ordinary or accelerated dynamics (accel=True).
        A total of M trajectories will be simulated and measurements are taken every thin_freq steps (to reduce the autocorrelation) after the first M*burnin_frac samples are
        rejected as burn in.
        Due to accumulating rounding errors, unitarity will be broken after some number of steps. To project back to the group manifold, all matrices are renormalised 
        every renorm_freq-th step. 
        The chain is fully defined (and thus reproducible) by the model and simulation parameters as well as the initial configuration of the chain and the state of the random number
        generator. By using the last configuration of a previous chain and the associated RNG state, one can continue the chain seamlessly in a new simulation.

        M: int
            number of HMC trajectories and thus total number of generated samples
        thin_freq: int
            frequency by which chain of generated samples will be thinned
        burin_frac: float
            fraction of total HMC samples needed for the system to thermalize  
        renorm_freq: int
            after how many trajectories are all matrices renormalized. Set to None to never renormalize
        accel: bool
            By default True, indicating to use Fourier Acceleration
        starting_config: (N,N,4)
            first configuration of the chain. If not passed a disordered (i.e hot) start will be used.
        RGN_state: str
            relative path to a .obj file containing the internal state of the random number generator from a previous run. 
            When using the final configuration of that run as the starting configuration for this one, the chain is seamlessly continued.

        measurements: list of callables
            can select from:
                internal_energy_density, susceptibility, ww_correlation_func
        chain_paths: list of str
            same size and order as measurements (if saving_bool=True, otherwise can be left empty), giving the file path relative to root folder to store the measurements.
            Do not include file extension, data is always saved as .npy file.
        saving_bool: bool
            save measurement data
        partial save: int
            after how many steps preliminary measurements and chain state is saved to disk. Requires saving_bool=True
        '''
        def saving(j, data, file_paths):
            '''Saving measurement chains and state of chain to potentially continue the current chain in a later run

            j: int
                number of measurements made
            data: list of np arrays
                each element is a chain of measurements, which might be (j,) or (j,n) (measuring scalar quantity or array of length n j times)
            file_paths: list of str
                relative file paths of measurements in the order that they appear in data
            '''
            # store measurement data collected so far at passed file paths
            for k,file_path in enumerate(file_paths):
                # make directory if it doesn't exist
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)
                np.save(file_path, data[k][:j+1])
        
            # store chain state
            os.makedirs('data/chain_state', exist_ok=True)
            with open('data/chain_state/RGN_state.obj', 'wb') as f:
                dump(np.random.get_state(), f)
            np.save('data/chain_state/config.npy', phi)

            return

        # np.random.seed(42) # for debugging
        if RGN_state is not None:
            with open(RGN_state, 'rb') as f:
                np.random.set_state(load(f))
        
        # take measurements and count accepted candidates after burn in
        start_id = int(np.floor(M*burnin_frac)) # number of steps taken in chain before measurements being
        self.sweeps = np.arange(M)[start_id+1::thin_freq] # positions in the chain when measurements were made
        self.M = self.sweeps.size # number of measurements

        # initialize arrays to store chain of measurements 
        data = []
        if self.internal_energy_density in measurements:
            data.append(np.zeros(self.M))
        if self.susceptibility in measurements:
            data.append(np.zeros(self.M))
        if self.ww_correlation_func in measurements:
            data.append(np.zeros((self.M, self.N))) # unprocessed (not normalized, on range [0,N)) wall to wall correlation function for each sample


        t1 = time.time()
        if starting_config is None:
            # # cold/ordered start
            # a0 = np.ones((self.N,self.N,1))
            # ai = np.zeros((self.N,self.N,3))
            # phi = np.concatenate([a0,ai], axis=2)

            # # hot start
            # sampling 4 points form 4D unit sphere assures that norm of parameter vector is 1 to describe SU(2) matrices. Use a spherically symmetric distribution such as gaussian
            a = np.random.standard_normal((self.N,self.N,4))
            phi = SU2.renorm(a)
        else:
            phi = starting_config

        if accel:
            self.A = self.kernel_inv_F()

        # construct chain and take measurements
        n_acc = 0 # number of accepted candidates
        j = 0 # counter for position in chain of measurements
        for i in range(M):
            # phi is most recent accepted sample, phi_new is a candidate sample.
            # if candidate gets accepted: phi -> phi_new, otherwise phi -> phi
            if renorm_freq is not None:
                if i % renorm_freq == 0:
                    SU2.renorm(phi)

            # take one step in the chain
            if accel: 
                pi = self.pi_samples()
                phi_new, pi_new = self.leapfrog_FA(phi, pi)
                delta_H = self.Ham_FA(phi_new,-pi_new) - self.Ham_FA(phi,pi)

            else:
                # the conjugate momenta are linear combination of Pauli matrices and thus described by 3 parameters
                pi = np.random.standard_normal((self.N,self.N,3))
                phi_new, pi_new = self.leapfrog(phi, pi)
                delta_H = self.Ham(phi_new,-pi_new) - self.Ham(phi,pi)

            acc_prob = np.min([1, np.exp(-delta_H)])

            if acc_prob > np.random.random():
                phi = phi_new
                if i > start_id:
                    n_acc += 1 
            else:
                phi = phi 

            # take measurements after burin and every thin_freq-th step in chain
            if (i > start_id) and (i-(start_id+1))%thin_freq == 0:
                for k, func in enumerate(measurements):
                    data[k][j] = func(phi)
                j += 1

            # store partial results of measurements and current state of chain
            if saving_bool and i>0 and i%partial_save == 0:
                print('saving partial results')
                saving(j, data, chain_paths)


        # store full chain of measurements and current state of chain
        if saving_bool:
            saving(self.M, data, chain_paths)

        self.acc_rate = n_acc/(M-start_id) # acceptance rate as decimal number
        self.time = time.time() - t1 # combined simulation and measurement time in seconds
        print('Completed %d steps in %s. \nSimulation rate: %.2f steps/sec \nAcceptance rate: %.2f%%'%(M, str(timedelta(seconds=self.time)), M/self.time, self.acc_rate*100))
        
        return
        # stop here and do ensemble average in individual files, allowing for flexibility and immediate plotting
        # can keep get_avg_error as a function in this file to be imported into the analysis ones. 


        # get ensemble avg and error 
        # returns = []
        # for d in data:
        #     returns.append(self.get_avg_error(d))
        # e_avg, e_err = self.get_avg_error(es)
        # chi_avg, chi_err, chi_IAT, chi_IAT_err = self.get_avg_error(chis, get_IAT=True)

        # cor, cor_err = self.get_avg_error(cor_funcs)
        # plot correlation function and get correlation length
        # xi, xi_err, reduced_chi2 = self.cor_length(cor, cor_err)



    # possible measurements
    def internal_energy_density(self, phi):
        '''Computes the internal energy (action) per site of configuration phi.

        phi: (self.N, self.N, 4)
            configuration to perform computation on

        Returns:
        e: float
            action per site
        '''
        e = self.action(phi) / (-self.beta * 4 * self.N**2) # definition of internal energy in terms of the action

        return e
      

    def ww_correlation_func(self, phi):
        ''' Computes the NON-normalized wall to wall correlation as described in the report via the cross correlation theorem.

        phi: (self.N, self.N, 4)
            configuration to perform computation on

        Returns:
        ww_cor: (self.N, )
            wall to wall correlation evaluated for wall separation in interval [0, self.N). NOT normalized to value at zero separation.
        '''
        ww_cor = np.zeros(self.N)
        Phi = np.sum(phi, axis=0) # (N,4)
        for k in range(4):
            cf, _ = correlations.correlator(Phi[:,k], Phi[:,k])
            ww_cor += cf
        ww_cor *= 4/self.N**2

        return ww_cor


    def susceptibility(self, phi):
        '''
        Computes the susceptibility i.e. the average point to point correlation for configuration phi.
        As described in the report, this closely related to summing the wall to wall correlation function which can be computed efficiently via the cross correlation theorem.

        phi: (self.N, self.N, 4)
            configuration to perform computation on

        Returns:
        chi: float
            susceptibility of phi
        '''
        chi = np.sum( 1/2*self.ww_correlation_func(phi) )

        return chi


### ---------------- ### Model independent routines commonly used

# perform ensemble average and get error estimate
def get_avg_error(data, get_IAT=False):
    '''Accepts sequence of measurements of a single observable and returns the average and the standard error on the mean, corrected for autocorrelation in the chain.
    Optionally also returns integrated autocorrelation time and its error.
    Elements of data, refereed to as data_point, can be either floats or 1D arrays of length N. The returned average and error will be of the same shape as data_point.

    data: (self.M, ) or (self.M, N)
        set of measurements made during the chain
    get_IAT: bool
        set to True to return IAT and its error

    Returns:
    avg: data_point.shape i.e. float or (N, )
        ensemble average of data
    error: float or (N, )
        SEM with correction factor through the IAT applied
    IAT: float
        integrated autocorrelation time (IAT)
    IAT_err: float 
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
        ts, ACF, ACF_err, IAT[i], IAT_err[i], comp_time = correlations.autocorrelator(data[:,i])
        error[i] *= np.sqrt(IAT[i])

    # to return floats if data points were floats
    if N == 1:
        avg, error = avg[0], error[0]
        IAT, IAT_err = IAT[0], IAT_err[0]

    if get_IAT:
        return avg, error, IAT, IAT_err
    
    return avg, error


# fit correlation length 
def corlength(ww_cor, ww_cor_err, data_save_path='', plot_save_path='', make_plot=False, show_plot=True):
    ''' 
    Processes ensemble average of wall to wall correlation function and its error to produce a plot and fit the analytically expected form of the correlation function.
    Optionally can store data and plot.
    
    ww_cor: (N, )
        ensemble average of correlation function on a lattice of length N
    ww_cor_err: (N, )
        associated error for all possible wall separations
    data_save_path: str
        path at which the processed correlation function will be stored as an .npy file (extension not needed in path)
    make_plot: bool
        make plot of correlation function with fit
    show_plot: bool
        show the produced plot. Only has an effect if make_plot=True
    plot_save_path: str
        path at which the produced plot will be stored. File extension needed and .pdf recommended

    Returns
    cor_length: float
        fitted correlation length in units of the lattice spacing
    cor_length_err: float
        error in the fitted correlation length
    reduced_chi2: float
        chi-square per degree of freedom as a goodness of fit proxy
    '''
    def fit(d,xi):
        return (np.cosh((d-N_2)/xi) - 1) / (np.cosh(N_2/xi) - 1)

    # normalize and use periodic bcs to get correlation for wall separation of N to equal that of separation 0
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    ww_cor, ww_cor_err = np.concatenate((ww_cor, [ww_cor[0]])), np.concatenate((ww_cor_err, [ww_cor_err[0]]))

    # use symmetry about N/2 due to periodic bcs and mirror the data to reduce errors (effectively increasing number of data points by factor of 2)
    N_2 = int(ww_cor.shape[0]/2) 
    ds = np.arange(N_2+1) # wall separations covering half the lattice length
    cor = 1/2 * (ww_cor[:N_2+1] + ww_cor[N_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:N_2+1]**2 + ww_cor_err[N_2::-1]**2) / np.sqrt(2)

    # store processed correlation function data
    if data_save_path != '':
        dir_path = os.path.dirname(data_save_path) # make directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        np.save(data_save_path, np.row_stack([ds, cor, cor_err]))

    # perform the fit  
    mask = cor > 0 # fitting range
    popt, pcov = curve_fit(fit, ds[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    if make_plot:
        fig = plt.figure(figsize=(8,6))

        plt.errorbar(ds, cor, yerr=cor_err, fmt='.', capsize=2)
        ds_fit = np.linspace(0, ds[mask][-1], 500)
        plt.plot(ds_fit, fit(ds_fit,*popt), c='g', label='$\\xi = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(cor_length, cor_length_err, reduced_chi2))
        plt.yscale('log')
        plt.xlabel(r'wall separation $d$ [$a$]')
        plt.ylabel('wall wall correlation $C_{ww}(d)$')
        plt.legend(prop={'size':12}, frameon=True, loc='upper right') # location to not conflict with error bars
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
        if show_plot:
            plt.show()
        else:
            dir_path = os.path.dirname(plot_save_path) # make directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            fig.savefig(plot_save_path)
            plt.close() # for memory purposes

    return cor_length, cor_length_err, reduced_chi2