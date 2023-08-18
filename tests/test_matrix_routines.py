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
    Runs all tests 5 times with different seed and lattice size L.
    '''
    Ls = np.random.randint(2, 100, size=1)
    for seed,L in enumerate(Ls):
        np.random.seed(seed)
        A = np.random.random((L,L,4))
        B = np.random.random((L,L,4))
        test_trace(A)
        test_determinant(A)
        test_hermitian_conjugate(A)
        test_sum(A, B)
        test_product(A, B)


def test_trace(A=np.zeros((1,1,4))):
    '''
    Trace of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    tr = SU2.tr(A)

    # convert np.matrix element at each lattice site to an ndarray and assert hermitian conjugation
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert np.allclose( tr[i,j], np.trace(SU2.make_mats(A)[i,j].A).real )


def test_determinant(A=np.zeros((1,1,4))):
    '''
    Determinants of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    det = SU2.det(A)

    # convert np.matrix element at each lattice site to an ndarray and assert hermitian conjugation
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert np.allclose( det[i,j], np.linalg.det(SU2.make_mats(A)[i,j].A).real )


def test_hermitian_conjugate(A=np.zeros((1,1,4))):
    '''
    Hermitian conjugate of a matrix valued lattice.

    Parameters
    ----------
    A: ndarray
        parameters describing an SU(2) matrix at every lattice site
    '''
    hc_matrix = SU2.make_mats(SU2.hc(A))

    # convert np.matrix element at each lattice site to an ndarray and assert hermitian conjugation
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert np.allclose( hc_matrix[i,j].A, (SU2.make_mats(A)[i,j].H).A )


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

    # convert np.matrix element at each lattice site to an ndarray and assert sum
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert np.allclose(sum_matrix[i,j].A, A_matrix[i,j].A + B_matrix[i,j].A)


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
    # convert np.matrix element at each lattice site to an ndarray and assert sum
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert np.allclose( product_matrix[i,j].A, np.matmul(A_matrix[i,j],B_matrix[i,j]).A )


def product_speed_comparision():
    '''
    Compares speed at which product of two SU2 matrices can be computes using generic numpy matrix multiplication vs the special SU2 routines acting on the entire lattice at once.
    For better compatibility, prints the ratio of the average execution times.

    Execute script manually to run (not done by pytest).
    '''
    L = np.random.randint(40, 700) # range of lattice sizes used during main analysis
    N = 10 # measures execution time N times and uses average
    time_SU2, time_np = np.ones((2,N))

    for i in range(N):
        A = np.random.random((L,L,4))
        A_matrix = SU2.make_mats(A)
        B = np.random.random((L,L,4))
        B_matrix = SU2.make_mats(B)

        # SU2 routines
        t1 = time.time()
        product = SU2.dot(A,B)
        t2 = time.time()
        time_SU2[i] = t2-t1

        # numpy matrix multiplication
        t1 = time.time()
        for j in range(L):
            for k in range(L):
                product = np.matmul(A_matrix[j,k],B_matrix[j,k])
        t2 = time.time()
        time_np[i] = t2-t1
        print('Completed {:d}/{:d}'.format(i+1,N))

    avg_time_SU2 = np.mean(time_SU2)
    avg_time_np = np.mean(time_np)
    ratio = np.divide(avg_time_SU2, avg_time_np, out=np.zeros_like(avg_time_SU2), where=avg_time_np>1e-7)
    print('------\nRatio of SU2 routine time to numpy multiplication on a lattice with L={:d}:\n{:.3e}\n------'.format(L, ratio))
    return

product_speed_comparision()