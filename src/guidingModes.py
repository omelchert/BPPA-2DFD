import numpy as np


def boundStates((x,dx),k0,nx0,nb):
    """compute bound states for mode operator

    Implements mode operator (see Ref. [2]) following Ref. [1] and solves its 
    eigenproblem to yield effective refractive inices and guided modes as 
    eigenvalues and eigenvectors, respectively.

    Args:
        x (numpy array, ndim=1): discretized x domain
        dx (float): uniform mesh width
        k0 (float): free space wave number
        nx0 (numpy array, ndim=1): considered 1D refractive index profile
        nb (float): refractive index at infinity

    Notes:
        Mode operator is self-adjoined, therefore its eigenvalues are real.
        Real eigenvalues (i.e. effective refractive indices) mean lossless
        propagation of the respective eigenvectors (i.e. guided modes).

        Effective refractive indices that refer to actual guiding modes need to 
        be larger than the refractive index at infinity. This is required for 
        the guiding mode to be normalizable. In turn, this gives a criterion
        to filter for allowed eigenvalues/eigenvector pairs.

    Returns:
        TE (list of tuples): sequence of sorted (effectiveRefractiveIndes,
            guidingMode) pairs, ordered for increasing values of the first
            argument. Thus, the eigenvalue/eigenvector pair of lowest order
            can be found at the zeroth array position.

    Refs:
        [1] Dielectric Waveguides
            Hertel, P.
            Lecture Notes: TEDA Applied Physiscs School

        [2] Modified finite-difference beam-propagation method based on the
            Douglas scheme
            Sun, L. and Yip, G. L.
            Optics Letters, 18 (1993) 1229
    """
    # IMPLEMENTS MODE OPERATOR FOR TE POLARIZED FIELD - EQ. (2.7) OF REF. [1]
    modeOperatorTE = np.diag(-2.0*np.ones(x.size)/dx/dx/k0/k0 + nx0*nx0, 0) +\
        np.diag(np.ones(x.size-1)/dx/dx/k0/k0 , 1) +\
        np.diag(np.ones(x.size-1)/dx/dx/k0/k0 , -1)

    # BEARING IN MIND THAT MODE OPERATOR IS SELF-ADJOINT, AN ALGEBRAIC SOLUTION
    # PROCEDURE FOR HERMITIAN MATRICES (NUMPYS EIGH) CAN BE EMPLOYED 
    eigVals, eigVecs = np.linalg.eigh(modeOperatorTE)

    # FILTER FOR ALL EIGENVALUES THAT ARE LARGER THAN THE REFRACTIVE INDEX
    # AT INFINITY. OTHERWISE, THE RESPECTIVE EIGENVECTORS WOULD BE OF 
    # OSCILLATING TYPE AT INFITY, PREVENTING NORMALIZATION
    TEList = []
    myFilter = eigVals > nb*nb
    for i in range(eigVals.size):
      if myFilter[i]:
        fac = 1./np.sqrt(dx) if eigVecs[:,i].sum() > 0 else -1/np.sqrt(dx) 
        TEList.append((np.sqrt(eigVals[i]),fac*eigVecs[:,i]))

    return sorted(TEList,key=lambda x: x[0],reverse=True) 


def dumpModes(x,TE):
    """dump modes to stdout

    Implements routine to inspect the found guiding modes

    Args:
        x (numpy array, ndim=1): discretized x domain
        TE (list of tuples): sequence of sorted (effectiveRefractiveIndes,
            guidingMode) pairs, ordered for increasing values of the first
            argument. Thus, the eigenvalue/eigenvector pair of lowest order
            can be found at the zeroth array position.

    """
    nModes = len(TE)

    for j in range(nModes):
        nEff,E = TE[j]
        print "# nEff_%d = %lf\n"%(j,nEff)

    for i in range(x.size):
        print x[i],
        for j in range(nModes):
            print TE[j][1][i],
        print


# EOF: guidingModes.py
