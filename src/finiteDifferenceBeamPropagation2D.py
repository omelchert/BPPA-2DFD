import numpy as np
from tridiagonalMatrixSolver import *
from boundaryConditions import *


def Chung1990Solver(((x,dx),(z,dz)),(nxz,n0,k0),E0,callbackFunc):
    """2D finite difference beam propagation method for paraxial wave equation 

    Implements 2D finite difference beam propagation algorithm for the paraxial
    wave equation presented by Ref. [1] with transparent boundary conditions 
    derived by Ref. [2]
        
    Args:
        (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
        (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
        nxz (numpy array, ndim=2): refractive index profile in x,z plane
        n0 (float): effective reference index
        k0 (float): wave number 
        E0 (numpy array, ndim=1): initial field profile
        callbackFunc (function): callback function performing measurements

    Returns:
        u (numpy array, ndim=1): updated field values at current z-step

    Notes:
        Ref. [4] refers to the implemented approximation by the term "paraxial 
        wave equation"

        Algorithm is based on repeatedly solving a tridiagonal system of linear
        equations in order to update field values on a discrete 1D grid. 
        The linear system of equations has a particularly simple form so that a 
        simplified version of the general tridiagonal solver of Ref. [3] can
        be employed.

        callback function (callee callbackFunc) takes 5 parameters in the form
        callbackFunc(n, (x,dx), (z,dz), E1, E), where:

            n (int): current iteration step
            (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
            (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
            E1 (numpy array, ndim=1): previous field profile
            E (numpy array, ndim=1): current field profile
        
    Refs:
        [1] An Assessment of Finite Difference Beam Propagation Method
            Chung, Y. and Dagli, N.
            IEEE J. Quant. Elect., 26 (1990) 1335

        [2] Transparent boundary condition for beam propagation
            G. R. Hadley
            Optics Letters, 16 (1991) 624

        [3] Numerical Recipes in Fortran 77
            Press, WH and Teukolsky, SA and Vetterling, WT and Flannery, BP
            Cambridge University Press (2nd Edition, 2002)

        [4] Finite-difference beam propagation method for guide-wave optics
            Xu, C. L. and Huang, W. P.
            PIER, 11 (1995) 1-49

    """
    # FIELD VALUES AT PREVIOUS AND CURRENT Z-STEP
    E1 = np.array(np.copy(E0), dtype=complex)
    E = np.zeros(E0.size, dtype=complex)

    # DIAGONAL AND OFF-DIAGONAL ELEMENTS IN THE TRIDIAGONAL SYSTEM DERIVED IN
    # EQ. (6) OF REF. [1]. LINES BELOW IMPLEMENT EQS. (6A-C), RESPECTIVELY
    a =  dz/dx/dx/2                                    
    b =  dz/dx/dx - dz/2*(nxz*nxz-n0*n0) + 2.*1j*k0*n0
    c = -dz/dx/dx + dz/2*(nxz*nxz-n0*n0) + 2.*1j*k0*n0

    for n in range(z.size-1):
        
        # UPDATE FIELD VALUES USING DEDICATED TRIDIAGONAL SOLVER
        E = CODiSolver(a,b[:,n+1],c[:,n],E1,E)
        # IMPOSE TRANSPARENT BOUNDARY CONDITIONS DERIVED BY REF. [2]
        E[-1] = TBC2D(dx, (E1[-1], E1[-2]), (E[-2],E[-1]))
        E[0]  = TBC2D(dx, (E1[0],  E1[1]),  (E[1], E[0]))

        # CALLBACKFUNC OBSERVABLES
        callbackFunc(n,(x,dx),(z,dz),E0,E)

        # ADVANCE Z-STEP
        E1[:] = E[:]


def Sun1993Solver(((x,dx),(z,dz)),(nxz,n0,k0),E0,callbackFunc,w=0.5):
    """2D finite difference beam propagation method for paraxial wave equation 

    Implements 2D finite difference beam propagation algorithm for the paraxial
    wave equation presented by Ref. [1] with transparent boundary conditions 
    derived by Ref. [2]
        
    Args:
        (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
        (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
        nxz (numpy array, ndim=2): refractive index profile in x,z plane
        n0 (float): effective reference index
        k0 (float): wave number 
        E0 (numpy array, ndim=1): initial field profile
        callbackFunc (function): callback function performing measurements

    Returns:
        u (numpy array, ndim=1): updated field values at current z-step

    Notes:
        Algorithm is based on repeatedly solving a tridiagonal system of linear
        equations, using the solver in Ref. [3], in order to update field 
        values on a discrete 1D grid. 

        callback function (callee callbackFunc) takes 5 parameters in the form
        callbackFunc(n, (x,dx), (z,dz), E1, E), where:

            n (int): current iteration step
            (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
            (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
            E1 (numpy array, ndim=1): previous field profile
            E (numpy array, ndim=1): current field profile
        
    Refs:
        [1] Modified finite-difference beam-propagation method based on the 
            Douglas scheme
            Sun, L. and Yip, G. L.
            Optics Letters, 18 (1993) 1229

        [2] Transparent boundary condition for beam propagation
            G. R. Hadley
            Optics Letters, 16 (1991) 624

        [3] Numerical Recipes in Fortran 77
            Press, WH and Teukolsky, SA and Vetterling, WT and Flannery, BP
            Cambridge University Press (2nd Edition, 2002)
    """
    # FIELD VALUES AT PREVIOUS AND CURRENT Z-STEP
    E1 = np.array(np.copy(E0), dtype=complex)
    E = np.zeros(E0.size, dtype=complex)
    # AUXILIARY SOLUTION VECTOR FOR TRIDIAGONAL LINEAR SYSTEM
    r = np.zeros(E0.size, dtype=complex)

    # DIAGONAL AND OFF-DIAGONAL ELEMENTS IN THE TRIDIAGONAL SYSTEM DERIVED IN
    # EQS. (4) OF REF. [1]. LINES BELOW IMPLEMENT COEFFICIENT MATRICES 
    # FOLLOWING EQ. (4). NOTE THAT FULL MATRIX REPRESENTATION IS USED BELOW
    nu = k0*k0*(nxz*nxz-n0*n0)
    aLHS = 1./12 + 1j*w*dz/2/n0/k0*(1./dx/dx + nu/12)
    bLHS = 5./6  - 1j*w*dz/n0/k0*(1./dx/dx - 5.*nu/12)
    aRHS = 1./12 - 1j*(1.-w)*dz/2/n0/k0*(1./dx/dx + nu/12)
    bRHS = 5./6  + 1j*(1.-w)*dz/n0/k0*(1./dx/dx - 5.*nu/12)

    for n in range(z.size-1):

        # COMPUTE RIGHT-HAND-SIDE OF TRIDIAGONAL SYSTEM 
        r[:] = bRHS[:,n]*E1[:]
        r[1:] +=aRHS[:-1,n]*E1[:-1]
        r[:-1] += aRHS[1:,n]*E1[1:]
        
        # UPDATE FIELD VALUES USING FULL TRIDIAGONAL SOLVER
        E = NumRecSolver(aLHS[:,n+1],bLHS[:,n+1],aLHS[:,n+1],r,E)

        # IMPOSE TRANSPARENT BOUNDARY CONDITIONS DERIVED BY REF. [2]
        E[-1] = TBC2D(dx, (E1[-1], E1[-2]), (E[-2],E[-1]))
        E[0]  = TBC2D(dx, (E1[0],  E1[1]),  (E[1], E[0]))

        # CALLBACKFUNC OBSERVABLES
        callbackFunc(n,(x,dx),(z,dz),E0,E)

        # ADVANCE Z-STEP
        E1[:] = E[:]


def Yamauchi1996Solver(((x,dx),(z,dz)),(nxz,n0,k0),E0,callbackFunc):
    """2D finite difference beam propagation method for paraxial wave equation 

    Implements 2D finite difference beam propagation algorithm for the paraxial
    wave equation presented by Ref. [1] with Dirichlet boundary conditions 
        
    Args:
        (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
        (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
        nxz (numpy array, ndim=2): refractive index profile in x,z plane
        n0 (float): effective reference index
        k0 (float): wave number 
        E0 (numpy array, ndim=1): initial field profile
        callbackFunc (function): callback function performing measurements

    Returns:
        u (numpy array, ndim=1): updated field values at current z-step

    Notes:
        Algorithm is based on repeatedly solving a tridiagonal system of linear
        equations, using the solver in Ref. [2], in order to update field 
        values on a discrete 1D grid. 

        callback function (callee callbackFunc) takes 5 parameters in the form
        callbackFunc(n, (x,dx), (z,dz), E1, E), where:

            n (int): current iteration step
            (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
            (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
            E1 (numpy array, ndim=1): previous field profile
            E (numpy array, ndim=1): current field profile
        
    Refs:
        [1] Improved Finite-Difference Beam Propagation-Method Based on the 
            Generalized Douglas Scheme and Its Application to Semivectorial 
            Analysis
            Yamauchi, J. and Shibayama, J. and Saito, O. and Uchiyama, O. and
            Nakano, H.
            Journal of Lightwave Technology, 14 (1996) 2401

        [2] Numerical Recipes in Fortran 77
            Press, WH and Teukolsky, SA and Vetterling, WT and Flannery, BP
            Cambridge University Press (2nd Edition, 2002)

    """
    # FIELD VALUES AT PREVIOUS AND CURRENT Z-STEP
    E1 = np.array(np.copy(E0), dtype=complex)
    E = np.zeros(E0.size, dtype=complex)
    # AUXILIARY SOLUTION VECTOR FOR TRIDIAGONAL LINEAR SYSTEM
    r = np.zeros(E0.size, dtype=complex)

    # DIAGONAL AND OFF-DIAGONAL ELEMENTS IN THE TRIDIAGONAL SYSTEM DERIVED IN
    # EQS. (2-5) OF REF. [1]. LINES BELOW IMPLEMENT COEFFICIENT MATRICES 
    # FOLLOWING EQ. (5). NOTE THAT FULL MATRIX REPRESENTATION IS USED BELOW
    nu = k0*k0*(nxz*nxz-n0*n0)
    zetaP = 1j*k0*n0/6 + dz/2/dx/dx + nu*dz/24
    zetaM = 1j*k0*n0/6 - dz/2/dx/dx - nu*dz/24
    xiP = 5*1j*k0*n0/3 - dz/dx/dx + 5*nu*dz/12
    xiM = 5*1j*k0*n0/3 + dz/dx/dx - 5*nu*dz/12

    for n in range(z.size-1):
        # COMPUTE RIGHT-HAND-SIDE OF TRIDIAGONAL SYSTEM 
        r[:] = xiP[:,n]*E1[:]
        r[1:] += zetaP[:-1,n]*E1[:-1]
        r[:-1] += zetaP[1:,n]*E1[1:]
        
        # UPDATE FIELD VALUES USING FULL TRIDIAGONAL SOLVER
        E = NumRecSolver(zetaM[:,n+1],xiM[:,n+1],zetaM[:,n+1],r,E)

        # IMPOSE DIRICHLET BOUNDARY CONDITIONS 
        #E[0] = 0
        #E[-1] = 0
        # IMPOSE TRANSPARENT BOUNDARY CONDITIONS DERIVED BY REF. [2]
        E[-1] = TBC2D(dx, (E1[-1], E1[-2]), (E[-2],E[-1]))
        E[0]  = TBC2D(dx, (E1[0],  E1[1]),  (E[1], E[0]))

        # CALLBACKFUNC OBSERVABLES
        callbackFunc(n,(x,dx),(z,dz),E0,E)

        # ADVANCE Z-STEP
        E1[:] = E[:]

        
# EOF: finiteDifferenceBeamPropagation2D.py
