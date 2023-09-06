# To ignore deprecation warnings, run
# pytest -W ignore::PendingDeprecationWarning
# Otherwise just
# pytest

import time
import numpy as np
from SU2xSU2 import SU2_mat_routines as SU2

import pytest 


def test_sequence():
    '''
    Runs all tests 5 times with varying lattice dimension D and size L.
    The lattice parameter are predefined to avoid issues of large memory allocation during the initialization 
    of a large and high dimensional array.
    '''
    Ds = [2, 3, 4, 5]
    Ls = [100, 35, 16, 10]
    for i in range(len(Ds)):
        np.random.seed(i)
        lattice_shape = tuple(np.repeat(Ls[i],Ds[i]))
        A = np.random.random(lattice_shape+(4,))
        B = np.random.random(lattice_shape+(4,))
        test_trace(A)
        test_determinant(A)
        test_hermitian_conjugate(A)
        test_sum(A, B)
        test_product(A, B)


def get_lattice_coords(lattice_shape):
    '''
    Computes a 2D array where each row gives the coordinates of a lattice site.

    Parameters
    ----------
    lattice_shape: tuple
        gives the shape of the cubic lattice, i.e. (L,...,L) with D occurrences of L.

    Returns
    -------
    lattice_coords: array
        array containing the coordinates of all lattice sites
    '''
    D = len(lattice_shape)
    L = lattice_shape[0]
    grid = np.indices(lattice_shape) 
    lattice_coords = np.reshape( np.moveaxis(grid, 0, -1), (L**D, D))

    return lattice_coords


def test_trace(A=np.zeros((1,1,4))):
    '''
    Trace of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    tr = SU2.tr(A)
    A_matrix = SU2.make_mats(A)
    lattice_shape = A.shape[:-1]
    lattice_coords = get_lattice_coords(lattice_shape)

    for coord in lattice_coords:
        mask = tuple(coord)
        assert np.allclose( tr[mask], np.trace(A_matrix[mask]).real )


def test_determinant(A=np.zeros((1,1,4))):
    '''
    Determinants of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    det = SU2.det(A)
    A_matrix = SU2.make_mats(A)
    lattice_shape = A.shape[:-1]
    lattice_coords = get_lattice_coords(lattice_shape)

    for coord in lattice_coords:
        mask = tuple(coord)
        assert np.allclose( det[mask], np.linalg.det(A_matrix[mask]).real )


def test_hermitian_conjugate(A=np.zeros((1,1,4))):
    '''
    Hermitian conjugate of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    hc_matrix = SU2.make_mats(SU2.hc(A))
    A_matrix = SU2.make_mats(A)
    lattice_shape = A.shape[:-1]
    lattice_coords = get_lattice_coords(lattice_shape)

    for coord in lattice_coords:
        mask = tuple(coord)
        assert np.allclose( hc_matrix[mask], A_matrix[mask].H )


def test_sum(A=np.zeros((1,1,4)), B=np.zeros((1,1,4))):
    '''
    Sum of two matrix values lattices

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    B: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    A_matrix = SU2.make_mats(A)
    B_matrix = SU2.make_mats(B)

    su2_sum, prop_const = SU2.sum(A,B)
    sum = prop_const*su2_sum
    sum_matrix = SU2.make_mats(sum)

    lattice_shape = A.shape[:-1]
    lattice_coords = get_lattice_coords(lattice_shape)

    for coord in lattice_coords:
        mask = tuple(coord)
        assert np.allclose( sum_matrix[mask], A_matrix[mask]+B_matrix[mask])


def test_product(A=np.zeros((1,1,4)), B=np.zeros((1,1,4))):
    '''
    Product of two matrix valued lattices.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    B: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    A_matrix = SU2.make_mats(A)
    B_matrix = SU2.make_mats(B)

    product = SU2.dot(A,B)
    product_matrix = SU2.make_mats(product)

    lattice_shape = A.shape[:-1]
    lattice_coords = get_lattice_coords(lattice_shape)

    for coord in lattice_coords:
        mask = tuple(coord)
        assert np.allclose( product_matrix[mask], np.matmul(A_matrix[mask],B_matrix[mask]) )


def product_speed_comparision():
    '''
    Compares speed at which product of two SU2 matrices can be computes using generic numpy matrix multiplication vs the special SU2 routines acting on the entire lattice at once.
    For better compatibility, prints the ratio of the average execution times.

    Execute script manually to run (not done by pytest).
    '''
    D = 4
    L = 16
    lattice_shape = tuple(np.repeat(L,D))
    N = 10 # measures execution time N times and uses average
    time_SU2, time_np = np.ones((2,N))

    for i in range(N):
        A = np.random.random(lattice_shape+(4,))
        A_matrix = SU2.make_mats(A)
        B = np.random.random(lattice_shape+(4,))
        B_matrix = SU2.make_mats(B)

        # SU2 routines
        t1 = time.time()
        product = SU2.dot(A,B)
        t2 = time.time()
        time_SU2[i] = t2-t1

        # numpy matrix multiplication
        t1 = time.time()
        lattice_shape = A.shape[:-1]
        lattice_coords = get_lattice_coords(lattice_shape)

        for coord in lattice_coords:
            mask = tuple(coord)
            product = np.matmul(A_matrix[mask],B_matrix[mask])
        t2 = time.time()
        time_np[i] = t2-t1
        print('Completed {:d}/{:d}'.format(i+1,N))

    avg_time_SU2 = np.mean(time_SU2)
    avg_time_np = np.mean(time_np)
    ratio = np.divide(avg_time_SU2, avg_time_np, out=np.zeros_like(avg_time_SU2), where=avg_time_np>1e-7)
    print('------\nRatio of SU2 routine time to numpy multiplication on a lattice with D={:d}, L={:d}:\n{:.3e}\n------'.format(D, L, ratio))
    return

product_speed_comparision()