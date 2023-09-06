# Several functions to perform common matrix operations for SU(2) matrices.
# By exploiting their properties, these routines are more efficient than general matrix methods
# Specifically, an SU(2) matrix is fully specified by 4 real parameters:
# [[a0 + i*a3, a2 + i*a1],
# [-a2 + i*a1, a0 - i*a3]]
#
# The operations are designed to act on entire SU(2) valued lattices.
# The lattice is assumed to be square but can be of arbitrary dimension and size.


import numpy as np


def alpha_to_a(alpha):
    '''
    A lattice of SU(2) matrices may be defined through the exponential map with parameters alpha. 
    The exponential map can be explicitly evaluated, resulting in a linear combination of the Pauli matrices and unity:
    U = exp(i alpha_i sigma_i) = a_0*1 + i*a_i sigma_i
    This function finds the coefficients ``a`` based on ``alpha``.    
    
    Parameters
    ----------
    alpha: (lattice shape,3) array
        parameters when representing a SU(2) group element via the exponential map at every lattice site

    Returns
    -------
    a: (lattice shape,4) array
        parameters of matrices at each lattice site when explicitly evaluating the exponential map
    '''
    lattice_shape = alpha.shape[:-1] # gives tuple (L,...,L)
    a = np.empty(lattice_shape+(4,))
    norm = np.sqrt(np.sum(alpha**2, axis=-1)) # (L,...,L)
    # to do arithmetic with another (L,...,L,3) array need to broadcast to include axis 2
    alpha_norm = norm.reshape(lattice_shape+(1,))
    # To avoid division by zero: if alpha_norm is 0, then alpha must be zero, such that the normalized alpha must be zero too
    alpha_unit = np.divide(alpha, alpha_norm, out=np.zeros_like(alpha), where=alpha_norm!=0)
    a[...,0] = np.cos(norm)
    a[...,1:] = alpha_unit * np.sin(alpha_norm)

    return a


def make_mats(a):
    '''
    Constructs explicit matrices corresponding to parameter vector ``a``.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    mats: (lattice shape) object array
        np.matrix instance at every site
    '''
    lattice_shape = a.shape[:-1] # gives tuple (L,...,L)
    D = len(lattice_shape)
    L = lattice_shape[0]
    # make 2D array where each row gives the coordinates of a lattice site to later easily loop over all lattice sites with their coordinates
    grid = np.indices(lattice_shape) 
    lattice_coords = np.reshape( np.moveaxis(grid, 0, -1), (L**D, D))
    # array containing np.matrix objects
    mats = np.empty(lattice_shape, dtype=object)

    for coord in lattice_coords:
        mask = tuple(coord)
        paras = a[mask] # (4,) returns parameter vector at lattice site specified by coord
        mat = [[paras[0]+1j*paras[3], paras[2]+1j*paras[1]], 
                [-paras[2]+1j*paras[1], paras[0]-1j*paras[3]]]
        mats[mask] = np.matrix(mat)

    return mats


### ----------------------- ###
### basic matrix quantities ###
### ----------------------- ###
def hc(a):
    '''
    Returns the parameter vector of the hermitian conjugate at each lattice site.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    a_hc: (lattice shape,4) array
        parameters of hermitian conjugate SU(2) valued lattice
    '''
    a_hc = -a 
    a_hc[...,0] = a[...,0]

    return a_hc


def tr(a):
    '''
    Returns the trace of the matrices at each lattice site.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    trace: (lattice shape) array 
        trace at each site of the SU(2) valued lattice
    '''
    trace = 2*a[...,0]
    return trace


def det(a):
    '''
    The determinant of an SU(2) matrix is given by the squared length of the parameter vector.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    determinant: (lattice shape) array
        determinants of the SU(2) valued lattice
    '''
    determinant = norm2(a)
    return determinant


def norm2(a):
    '''
    Returns squared norm of the parameter vector ``a``.
    
    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    norm_sq: (lattice shape) array 
        containing the norm at each site
    '''
    norm_sq = np.sum(a**2, axis=-1)
    return norm_sq


def renorm(a):
    '''
    Renormalises matrix to have det = 1

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of the SU(2) valued lattice

    Returns
    -------
    renormed: (lattice shape,4) array
        renormalised parameters of the SU(2) valued lattice
    '''
    lattice_shape = a.shape[:-1] # gives tuple (L,...,L)
    norm =  np.sqrt(norm2(a)).reshape(lattice_shape+(1,)) # broadcast to do arithmetic with (lattice shape,4) array a
    renormed = np.divide(a, norm, out=np.zeros_like(a), where=norm!=0)
    
    return renormed


### ---------------------------- ###
### combining two SU(2) matrices ### 
### ---------------------------- ###
def dot(a, b):
    '''
    Computes the elementwise matrix product between two lattices of SU(2) matrices with parameter vectors ``a`` and ``b``.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of first SU(2) valued lattice
    b: (lattice shape,4) array
        parameters of second SU(2) valued lattice

    Returns
    -------
    c: (lattice shape,4) array
        parameters of SU(2) valued lattice resulting from the elementwise matrix products   
    '''
    c = np.empty_like(a)
    c[...,0] = a[...,0]*b[...,0] - np.sum(a[...,1:]*b[...,1:], axis=-1)
    c[...,1] = a[...,0]*b[...,1] + a[...,1]*b[...,0] + a[...,3]*b[...,2] - a[...,2]*b[...,3]
    c[...,2] = a[...,0]*b[...,2] + a[...,2]*b[...,0] + a[...,1]*b[...,3] - a[...,3]*b[...,1]
    c[...,3] = a[...,0]*b[...,3] + a[...,3]*b[...,0] + a[...,2]*b[...,1] - a[...,1]*b[...,2]

    return c


def sum(a, b):
    '''
    Computes the elementwise sum of two SU(2) valued lattices A and B with parameters ``a`` and ``b``.
    Let C = A + B, i.e. c = a + b. 
    Note that the sum of two SU(2) matrices is proportional to an SU(2) matrix with proportionality constant ``k``, meaning
    D = C/k = 1/k (A + B) is in SU(2).
    To only having to perform manipulations on SU(2) matrices, the parameters ``d`` of the SU(2) valued lattice D and the 
    constant ``k`` are returned such that their product gives the 
    parameter vectors of C, the sum of lattice A and B.

    Parameters
    ----------
    a: (lattice shape,4) array
        parameters of first SU(2) valued lattice
    b: (lattice shape,4) array
        parameters of second SU(2) valued lattice

    Returns
    -------
    d: (lattice shape,4) array
        parameters of SU(2) valued lattice proportional to a+b
    k: (lattice shape,1) array
        proportionality constant between d and a+b

    '''
    c = a + b
    k2 = 2*(a[...,0]*b[...,0] + np.sum(a[...,1:]*b[...,1:], axis=-1) + 1) # (L,...,L)
    lattice_shape = a.shape[:-1] # gives tuple (L,...,L)
    k = np.sqrt(k2, dtype=complex).reshape(lattice_shape+(1,))
    d = c / k

    return d, k