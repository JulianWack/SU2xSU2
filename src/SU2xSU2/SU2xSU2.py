import numpy as np
from pickle import dump, load 
import os
import time
from datetime import timedelta

from . import SU2_mat_routines as SU2
from .correlations import correlator


class SU2xSU2():
    '''Each instance describes a realization of the SU(2)xSU(2) model on a square lattice whose dynamics can be simulated using the 
    Fourier Accelerated Hybrid Monte Carlo algorithm.'''
    def __init__(self, D, L, a, ell, eps, beta, mass=0.1): 
        '''
        Parameters
        ----------
        D: int
            dimension of lattice
        L: int
            Number of lattice sites along one dimension. Must be even for implementation of Fourier acceleration to work properly 
        a: float
            Lattice spacing
        ell: int
            Number of steps to integrate Hamilton's equation, each of size eps
        eps: float
            Step size for integrating Hamilton's equations
        beta: float
            Model parameter defined in terms of the nearest neighbor coupling parameter g via beta = 1/(2g^2)
        mass: float (optional)
            Mass parameter used in Fourier Acceleration. The physics is independent of the parameter value which only effects the simulation performance.
            The most efficient choice depends on beta.
        '''
        # lattice parameters
        self.D, self.L, self.a = int(D), int(L), a
        self.lattice_shape = tuple(np.repeat(self.L,self.D))
        # leapfrog parameters
        self.ell, self.eps = int(ell), eps
        # model parameters
        self.beta = beta
        # acceleration parameters
        self.mass = mass

        # find mask to index the SU(2) valued field phi, giving the parameters of the nearest neighbors along the D positive directions in the lattice, followed by the negative ones
        self.NN_mask = self.make_NN_mask() 


    def make_NN_mask(self):
        '''Makes mask to apply to SU(2) valued field phi or momentum field pi which then gives the matrix parameter values of the nearest neighbors (NN) for each lattice site.
        Hence ``phi[self.NN_mask]`` is of shape (lattice shape,#neighbors,#parameters) i.e (L, ..., L, 2*D, 4) with D occurrences of L.
        
        Returns
        -------
        NN_mask: tuple
            tuple of D (lattice shape, 2*D, 1) arrays, each giving the coordinate along one axis for the nearest neighbors
        '''
        # make array containing the the coordinates of each lattice site 
        grid = np.indices(self.lattice_shape) 
        lattice_coords = np.moveaxis(grid, 0, -1) # (lattice shape, D) D-dimensional coordinates of each lattice site

        # make array where each element contains the nearest neighbor (NN) coordinates of the element 
        # the first D nearest neighbors are those along the positive directions of the lattice, while the later D neighbors are along negative direction 
        NN_shape = self.lattice_shape+(int(2*self.D),self.D) # (lattice shape, number of NN, D)
        NN = np.empty(NN_shape, dtype=int) # contains coords of NN for each lattice site

        for ax in range(self.D):
            # using ... for variable lattice shape
            # np.roll takes care of periodic boundary conditions and with a step of -1 elements are moved one unit along the positive direction of the axis
            NN[...,ax,:] = np.roll(lattice_coords, -1, axis=ax) # neighbor coordinates along positive direction of current axis
            NN[...,self.D+ax,:] = np.roll(lattice_coords, 1, axis=ax) # neighbor coordinates along negative direction of current axis

    	# make mask to easily index the field, yielding the nearest neighbors
        NN_mask_list = []
        for idx in range(self.D):
            NN_mask_list.append(NN[...,idx])

        NN_mask = tuple(NN_mask_list)

        return NN_mask

        
    def action(self, phi):
        '''
        Computes the action for lattice configuration ``phi``.

        Parameters
        ----------
        phi: (lattice shape, 4) array
            parameters of the SU(2) valued field at each lattice site

        Returns
        -------
        S: float
            the action
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask] # (L,...,L,2*D,4): containing the 4 paras of each of the 2*D NN

        # sum over lattice unit vectors i.e. positive directions only: to the right and up. Hence only need right and top NN, stored at position 0,3 respectively
        G = np.zeros(self.lattice_shape)
        for i in range(self.D):
            A = SU2.dot(phi_hc, phi_NN[...,i,:])
            G += SU2.tr(A + SU2.hc(A)) # when getting UFuncTypeError, check that dtype of G and SU2.tr is the same (float64 by default)

        # sum over lattice sites    
        S = -1/2 * self.beta * np.sum(G)

        return S


    def Ham(self, phi, pi):
        '''
        Computes the Hamiltonian for a lattice configuration ``phi`` and momentum configuration ``pi``.
        The kinetic term is chosen such that the momenta follow a standard Gaussian.

        Parameters
        ----------
        phi: (lattice shape,4) array
            parameters of the SU(2) valued field
        pi: (lattice shape,3) array
            parameter values of the conjugate momenta at each lattice site
            
        Returns
        -------
        H: float
            the Hamiltonian as the sum of the action and the kinetic term
        '''
        T = 1/2 * np.sum(pi**2) # equivalent to first summing the square of the parameters at each site and then sum over all sites
        S = self.action(phi)
        H = T + S

        return H 


    def Ham_FA(self, phi, pi):
        '''
        Analogous to :py:meth:`SU2xSU2.SU2xSU2.Ham` but computes the modified Hamiltonian used to accelerate the dynamics.

        Parameters
        ----------
        phi: (L,L,4) array
            parameters of the SU(2) valued field
        pi: (L,L,3) array
            parameter values of the conjugate momenta at each lattice site
            
        Returns
        -------
        H: float
            the modified Hamiltonian as the sum of the action and the modified kinetic term
        
        See Also
        --------
        SU2xSU2.SU2xSU2.Ham
        '''
        # (L,L) find magnitude of FT of each component of momentum in Fourier space. Then sum over all 3 components
        pi_F_mag = np.sum( np.abs(np.fft.fft2(pi, axes=(0,1)))**2, axis=-1 ) 
        T = 1/(2*self.L**2) * np.sum(pi_F_mag*self.A) # sum over momentum Fourier lattice
        S = self.action(phi)
        H = T + S 

        return H


    def prod_A_pi(self, pi_F):
        '''Computes the elementwise product of the inverse action kernel in Fourier space (denoted as 'A') and the momenta in Fourier space.

        Parameters
        ----------
        pi_F: (L,L,3) array
            parameter vector of momenta in Fourier space

        Returns
        -------
        prod: (L,L,3)
            parameter vector of momenta in Fourier space, each site being weighted by the inverse Fourier space kernel
        '''
        prod = np.multiply(self.A.reshape((self.L,self.L,1)), pi_F)
        return prod


    def kernel_inv_F(self):
        '''Finds inverse of the action kernel computed in the Fourier space, here referred to as ``A``.

        Returns
        -------
        A: (L,L) array
            inverse action kernel in Fourier space
        '''
        # x = 0.9 # parameter interpolating between accelerated (x=1) and unaccelerated (x=0) case. 
        # Appropriate kernel: A[k,k_] = (1 - x/2 - x/4*(np.cos(np.pi*ks[k]/self.L) + np.cos(np.pi*ks[k_]/self.L)) )**(-1)
        ks = np.arange(0, self.L) # lattice sites in Fourier space along one direction
        A = np.zeros((self.L,self.L)) # inverse kernel computed at every site in Fourier space
        for k in range(self.L):
            for k_ in range(k,self.L):
                A[k,k_] = ( 4*np.sin(np.pi*ks[k]/self.L)**2 + 4*np.sin(np.pi*ks[k_]/self.L)**2 + self.mass**2)**(-1)   
                A[k_,k] = A[k,k_] # exploit symmetry of kernel under exchange of directions 

        return A


    def pi_dot(self, phi):
        '''Computes the derivative of ``pi`` with respect to the Hamiltonian time. From Hamilton's equations, this is given as i times the derivative of the action wrt. ``phi``.
        ``pi`` and its time derivative are linear combinations of the Pauli matrices and hence described by 3 real parameters alpha.

        Parameters
        ----------
        phi: (lattice shape,4) array
            parameters of the SU(2) valued field

        Returns
        -------
        pi_t: (lattice shape,3) array
            parameters of the time derivative of the conjugate momenta
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask]
        # need sum of NN pairs along the two lattice unit vectors i.e. right+left and top+bottom
        alpha = np.zeros(self.lattice_shape+(3,))
        for idx in range(self.D):
            # sum is proportional to SU2 matrix, allowing to apply the SU2 product routine once proportionality constant has been identified
            sum_in_SU2, prop_const = SU2.sum(phi_NN[...,idx,:], phi_NN[...,idx+self.D,:]) # both are potentially complex but their product is always real
            V = (prop_const * SU2.dot(sum_in_SU2, phi_hc)).real
            alpha += 2*V[...,1:] # 3 parameters describing matrix -i(V - V^dagger) for the currently considered direction in the lattice
        pi_t = self.beta * alpha

        return pi_t


    def exp_update(self, pi_dt):
        '''Computes the update matrix of the field for the transition t -> t + dt.
        The update matrix is given by the exponential of the conjugate momenta pi times i dt. 
        As the momentum field is a linear combination of the SU(2) generators, the update matrix is itself an SU(2) element, described by 
        the parameter vector of pi scaled by dt.

        Parameters
        ----------
        pi_dt: (lattice shape,4) array
            parameters of the momenta times the integration step size
        Returns
        -------
        update: (lattice shape,4) array
            parameters of the update matrices
        '''
        update = SU2.alpha_to_a(pi_dt)
        return update


    def leapfrog(self, phi_old, pi_old):
        '''
        Performs the leapfrog integration of Hamilton's equations for ``self.ell`` steps, each of size ``self.eps``. The passed arguments define the initial conditions. 
        
        Parameters
        ----------
        phi_old: (lattice shape,4) array
            last accepted sample in the chain of the SU(2) valued field
        pi_old: (lattice shape,3) array
            parameters of conjugate momenta associated with phi_old
            
        Returns
        -------
        phi_cur: (lattice shape,4) array
            SU(2) matrix parameters after simulating dynamics
        pi_cur: (lattice shape,3) array
            momenta parameters after simulating dynamics
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
        Leapfrog integrator analogous to :py:meth:`SU2xSU2.SU2xSU2.leapfrog` but using the modified EoMs.

        See Also
        --------
        SU2xSU2.SU2xSU2.leapfrog
        '''
        def pi_FA(pi):
            '''
            Computes the modified momentum term entering in the exponential update in the accelerated dynamics.
            The modified momentum is given by the ordinary momentum pi, weighted by the inverse kernel which is easiest computed in Fourier space.
            
            Parameters
            ----------
            pi: (L,L,4) array
                real space parameters of conjugate momenta
            
            Returns
            -------
            pi_mod: (L,L,4) array
                real space parameters of modified conjugate momenta
            '''
            pi_F = np.fft.fft2(pi, axes=(0,1))
            pi_mod = np.real( np.fft.ifft2(self.prod_A_pi(pi_F), axes=(0,1)) )
            return pi_mod

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
        '''Returns real space sample of momenta according to the distribution based on the modified kinetic term in the modified Hamiltonian.
        The sampling is easiest done in Fourier space and in terms of a real and hermitian object ``PI`` from which the momentum samples can be reconstructed (both in Fourier space) 
        The size of the lattice along one dimension L is assumed to be even.

        Returns
        -------
        pi: (L,L,3) array
            parameters for the sample of the conjugate momenta in real space
        '''
        # momenta in Fourier space
        pi_F = np.zeros((self.L, self.L, 3), dtype=complex)

        PI_std = np.sqrt(self.L**2 / self.A) 
        STD = np.repeat(PI_std[:,:,None], repeats=3, axis=2) # standard deviation is identical for components at same position
        PI = np.random.normal(loc=0, scale=STD) #  (L,L,3) as returned array matches shape of STD

        # assign special modes for which FT exponential becomes +/-1. To get real pi in real space, the modes must be real themselves.
        N_2 = int(self.L/2)
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


    def run_HMC(self, M, burnin_frac, thin_freq=1, accel=True, measurements=[], ext_measurement_shape=[], chain_paths=[], 
                saving_bool=True, partial_save=5000, starting_config_path='', RNG_state_path='',
                chain_state_dir='data/chain_state/', renorm_freq=10000):
        '''
        Performs the Hybrid Monte Carlo simulation, by default with Fourier Acceleration.
    
        A total of ``M`` trajectories will be simulated and measurements are taken every ``thin_freq`` steps after the
        thermalisation period defined as the first ``M*burnin_frac`` samples.
        Due to accumulating rounding errors, the molecular dynamics cause field to leave the group manifold slowly. This is counteracted by enforcing the unitarity constraint
        of the matrix parameters every ``renorm_freq``-th step.
        The Monte Carlo chain is fully defined (and thus reproducible) by the model and simulation parameters as well as the initial configuration of the chain and 
        the state of the random number generator (RNG). By using the last configuration of a previous chain and the associated RNG state,
        one can continue the chain seamlessly in a new simulation.

        Parameters 
        ----------
        M: int
            number of HMC trajectories and thus total number of generated samples
        burin_frac: float
            fraction of total HMC samples which are rejected as burn in  
        thin_freq: int (optional)
            frequency at which measurements will be taken
        accel: bool (optional)
            By default True, indicating to use Fourier Acceleration
        measurements: list of callables (optional)
            upon defining an instance of the SU2xSU2 class called 'model', can select from: model.internal_energy_density, model.susceptibility, model.ww_correlation_func 
            (measuring the internal energy density, susceptibility, and wall-to-wall correlation respectively) or pass an externally defined function in which case
            ``external_measurement_structure`` must be specified. 
            Externally defined functions are required to expected to only have two arguments: the field and momentum configuration respectively. 
        ext_measurement_shape: list of tuples (optional)
            Only required if at least one of the callables in ``measurements`` is externally defined.
            Gives the data shape of a single measurement made by an externally defined function, matching their sequence of occurrence in ``measurements``.
            If a scalar quantity is measured, enter ``()``. For a 1D array of length 'A' enter ``(A,)``, for a 2D array enter ``(A,B)`` etc. 
            The data shape for SU2xSU2 class methods should not be specified. 
        chain_paths: list of str (optional)
            Only required if ``saving_bool=True``, otherwise can be left empty. Listing the file paths relative to current working directory to store the measurements. 
            The path must match their sequence of occurrence in ``measurements``.
            The data will always be saved as a .npy file, allowing to omit the file extension.
        saving_bool: bool (optional)
            save measurement data
        partial_save: int (optional)
            after how many steps preliminary measurements and chain state is saved to disk. Requires ``saving_bool=True``.
        starting_config_path: str (optional)
            path to configuration to initialize the chain (.npy file). If not passed a disordered (i.e hot) start will be used.
        RNG_state_path: str (optional)
            relative path to a .obj file containing the internal state of the random number generator from a previous run. 
            When using the final configuration and the associated RNG state of a previous run as the starting configuration for a new one, the chain is seamlessly continued.
        chain_state_dir: str (optional)
            path to directory in which the RNG state and the last configuration will be saved  
        renorm_freq: int (optional)
            after how many trajectories the SU(2) valued fields are projected back to the group manifold. Set to ``None`` to never renormalize
        '''
        def saving(j, data, file_paths):
            '''
            Saving measurement chains and the RNG state of chain to potentially continue the current chain in a later run

            Parameters
            ----------
            j: int
                number of measurements made
            data: list of np arrays
                each element is a chain of measurements, which might be (j,) or (j,n) (measuring scalar quantity or array of length n j times)
            file_paths: list of str
                relative file paths of measurements in the order that they appear in data
            '''
            # store measurement data collected so far at passed file paths
            # first check if paths contains any directories and create those if they don't exist already
            for k,file_path in enumerate(file_paths):
                dir_path = os.path.dirname(file_path)
                if dir_path != '':
                    os.makedirs(dir_path, exist_ok=True)
                np.save(file_path, data[k][:j+1])
        
            # store chain state
            os.makedirs(chain_state_dir, exist_ok=True)
            with open(os.path.join(chain_state_dir, 'RNG_state.obj'), 'wb') as f:
                dump(np.random.get_state(), f)
            np.save(os.path.join(chain_state_dir, 'config.npy'), phi)

            return

        # np.random.seed(42) # for debugging
        if RNG_state_path != '':
            with open(RNG_state_path, 'rb') as f:
                np.random.set_state(load(f))
        
        # take measurements and count accepted candidates after burn in
        start_id = int(np.floor(M*burnin_frac)) # number of steps taken in chain before measurements being
        self.sweeps = np.arange(M)[start_id+1::thin_freq] # positions in the chain when measurements were made
        self.M = self.sweeps.size # number of measurements

        # initialize arrays to store chain of measurements 
        data = []
        ext_count = 0 # counter to keep track of externally defined measurement functions
        for measurement in measurements:
            if measurement in [self.internal_energy_density, self.susceptibility]:
                data.append(np.zeros(self.M))
            elif measurement in [self.ww_correlation_func]:
                data.append(np.zeros((self.M, self.L))) # unprocessed (not normalized, on range [0,L)) wall to wall correlation function for each sample
            else:
                if ext_measurement_shape == []:
                    raise Exception('Please supply the data shape for the externally defined measurement: %s.'%(measurement.__name__))
                shape = (self.M,) + ext_measurement_shape[ext_count]
                data.append(np.zeros(shape))
                ext_count += 1


        t1 = time.time()
        if starting_config_path == '':
            # # cold/ordered start
            # a0 = np.ones(self.lattice_shape+(1,))
            # ai = np.zeros(self.lattice_shape+(3,))
            # phi = np.concatenate([a0,ai], axis=-1)

            # # hot start
            # sampling 4 points form 4D unit sphere assures that norm of parameter vector is 1 to describe SU(2) matrices. 
            # Use a spherically symmetric distribution such as gaussian
            a = np.random.standard_normal(self.lattice_shape+(4,))
            phi = SU2.renorm(a)
        else:
            phi = np.load(starting_config_path)

        if accel:
            self.A = self.kernel_inv_F()

        # construct chain and take measurements
        n_acc = 0 # number of accepted candidates
        j = 0 # counter for position in chain of measurements
        for i in range(int(M)):
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
                pi = np.random.standard_normal(self.lattice_shape+(3,))
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
                    # when measurement is SU2xSU2 class method, only the field configuration is required
                    # for maximal flexibility of externally defined functions, pass the field and momentum configuration 
                    if func in [self.internal_energy_density, self.ww_correlation_func, self.susceptibility]:
                        data[k][j] = func(phi)
                    else:
                        data[k][j] = func(phi, pi_new)
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



    ### ----------------- ###
    # possible measurements #
    ### ----------------- ###

    def internal_energy_density(self, phi):
        '''
        Computes the internal energy (action) per site of a lattice configuration.

        Parameters
        ----------
        phi: (lattice shape,4) array
            lattice configuration to perform computation on

        Returns
        -------
        e: float
            action per site
        '''
        e = self.action(phi) / (-self.beta * 4 * self.L**self.D) # definition of internal energy in terms of the action

        return e
      

    def ww_correlation_func(self, phi):
        ''' 
        Computes the **non**-normalized wall to wall correlation function for the lattice configuration ``phi``.
        Observing that the correlation of two variables can be written as a convolution, one can apply the cross correlation theorem to efficiently compute
        the latter using FFTs.  
        A class of walls is described by their normal vector which will be one of the lattice unit vectors. Due to the symmetry of the lattice, all these classes
        are equivalent, allowing to compute D wall to wall correlation functions from a single lattice configuration. The returned correlation function is the average of these. 

        Parameters
        ----------
        phi: (lattice shape,4) array
            configuration to perform computation on

        Returns
        -------
        ww_cor: (L,) array
            wall to wall correlation evaluated for wall separation in interval [0, L), **not** normalized to value at zero separation.
        '''
        ww_cors = np.zeros((self.D,self.L))
        lattice_axes = np.arange(self.D)
        # sum over all but one of the spatial axes (0,...,D-1), corresponding to the one over which the correlation is computed
        for i in lattice_axes:
            ax =  np.delete(lattice_axes, i) # axes which are summed over
            Phi = np.sum(phi, axis=tuple(ax)) # (L,4)
            cor = np.zeros(self.L)
            for k in range(4):
                cf, _ = correlator(Phi[:,k], Phi[:,k])
                cor += cf
            cor *= 4/self.L**self.D
            ww_cors[i] = cor

        ww_cor = np.mean(ww_cors, axis=0)

        return ww_cor


    def susceptibility(self, phi):
        '''
        Computes the susceptibility (the average point to point correlation) for the configuration ``phi`` which can be obtained by summing up the correlation function.

        Parameters
        ----------
        phi: (lattice shape,4) array
            configuration to perform computation on

        Returns
        -------
        chi: float
            susceptibility of phi
        '''
        chi = np.sum( 1/2*self.ww_correlation_func(phi) )

        return chi