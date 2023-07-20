# Several functions to perform common matrix operations for SU(2) matrices.
# By exploiting their properties, these routines are more efficient than general matrix methods
# Specifically, an SU(2) matrix is fully specified by 4 real parameters:
# [[a0 + i*a3, a2 + i*a1],
# [-a2 + i*a1, a0 - i*a3]]
#
# The operations are designed to be applies to a N by N lattice of SU(2) matrices, i.e. to parameter array of shape (N,N,4).


import numpy as np


def alpha_to_a(alpha):
    '''Convert parameters of SU(2) matrix when given in terms of the exponential map to the parameters when given as linear combination of gens and unity:
    U = exp(i alpha_i sigma_i) = a_0*1 + i*a_i sigma_i
        
    alpha: (N,N,3) array
        parameters when representing a SU(2) group element via the exponential map at every lattice site

    Returns:
    a: (N,N,4) array
        parameters of matrices at each lattice site when explicitly evaluating the exponential map
    '''
    N = alpha.shape[0]
    a = np.empty((N,N,4))
    norm = np.sqrt(np.sum(alpha**2, axis=2)) # (N,N)
    # to do arithmetic with other (N,N,3) array need to broadcast to include axis 2
    alpha_norm = norm.reshape((N,N,1))
    alpha_unit = np.divide(alpha, alpha_norm, out=np.zeros_like(alpha), where=alpha_norm!=0) # avoids division by zero. When norm is zero, i.e alpha is zero, alpha_unit is set to zero too 
    a[:,:,0] = np.cos(norm)
    a[:,:,1:] = alpha_unit * np.sin(alpha_norm)

    return a


def make_mats(a):
    '''Constructs explicit matrices corresponding to parameter vectors a.

    a: (N,N,4) array
        parameters of the SU(2) matrices

    Returns:
    mats: (N,N) object array
        np.matrix instance at every site
    '''
    N = a.shape[0]
    mats = np.empty((N,N), dtype=object)

    for i in range(N):
        for j in range(N):
            paras = a[i,j,:]
            mat = [[paras[0]+1j*paras[3], paras[2]+1j*paras[1]], 
                [-paras[2]+1j*paras[1], paras[0]-1j*paras[3]]]
            mats[i,j] = np.matrix(mat)

    return mats


### basic quantities ###
def hc(a):
    '''Returns the parameter vector of the hermitian conjugate at each lattice site.

    a: (N,N,4) array
        parameters of matrices

    Returns:
    a_hc: (N,N,4) array
        parameters of hermitian conjugate matrices
    '''
    a_hc = -a 
    a_hc[:,:,0] = a[:,:,0]

    return a_hc


def tr(a):
    '''Returns trace of matrices at each lattice site

    a: (N,N,4) array
        parameters of matrices

    Returns: (N,N) array trace at each site 
    '''
    trace = 2*a[:,:,0]
    return trace


def det(a):
    '''determinant is given by the squared length of the parameter vector
    '''
    return norm2(a)


def norm2(a):
    '''Returns squared norm of parameter vector
    
    a: (N,N,4) array
        parameters of matrices

    Returns: (N,N) array containing the norm at each site
    '''
    return np.sum(a**2, axis=2)


def renorm(a):
    '''Renormalises matrix to have det = 1
    '''
    N = a.shape[0]
    norm =  np.sqrt(norm2(a)).reshape((N,N,1)) # broadcast to do arithmetic with (N,N,4) array a
    renormed = np.divide(a, norm, out=np.zeros_like(a), where=norm!=0)
    
    return renormed


### combining two SU(2) matrices ### 
def dot(a, b):
    '''Computes matrix product A.B when matrices A and B have associated parameter vectors a and b.

    a,b: (N,N,4) array
        parameters of matrices at each lattice site

    Returns:
    c: (N,N,4) array
        parameters of matrix products   
    '''
    c = np.empty_like(a)
    c[:,:,0] = a[:,:,0]*b[:,:,0] - np.sum(a[:,:,1:]*b[:,:,1:], axis=2)
    c[:,:,1] = a[:,:,0]*b[:,:,1] + a[:,:,1]*b[:,:,0] + a[:,:,3]*b[:,:,2] - a[:,:,2]*b[:,:,3]
    c[:,:,2] = a[:,:,0]*b[:,:,2] + a[:,:,2]*b[:,:,0] + a[:,:,1]*b[:,:,3] - a[:,:,3]*b[:,:,1]
    c[:,:,3] = a[:,:,0]*b[:,:,3] + a[:,:,3]*b[:,:,0] + a[:,:,2]*b[:,:,1] - a[:,:,1]*b[:,:,2]

    return c


def sum(a, b):
    '''Computes sum of two SU(2) matrix lattices A and B with parameters a and b.
    Let C = A + B, i.e. c = a + b. 
    Note that the sum of two SU(2) matrices is proportional to an SU(2) matrix with proportionality constant k, meaning
    D = C/k = 1/k (A + B) is in SU(2).
    To only having to perform manipulations on SU(2) matrices, the parameters d of the SU(2) matrix lattice D and the constant k is returned such that their product gives the 
    parameter vectors of C, the sum of lattice A and B.

    a,b: (N,N,4) array
        parameters of matrices to sum at each lattice site

    Returns:
    d: (N,N,4) array
        parameters of SU(2) matrices
    k: (N,N,1) array
        proportionality constant between d and sum of matrices

    '''
    c = a + b
    k2 = 2*(a[:,:,0]*b[:,:,0] + np.sum(a[:,:,1:]*b[:,:,1:], axis=2) + 1) # (N,N)
    N = a.shape[0]
    k = np.sqrt(k2, dtype=complex).reshape((N,N,1))
    d = c / k

    return d, k