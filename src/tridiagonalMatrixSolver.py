import numpy as np


def CODiSolver(a,b,c,u1,u):
    """tridiagonal matrix solver for field update

    Implements tridiagonal matrix solver, adapted to the system of linear 
    equations for the field update in 2D beam propbagation presented by 
    Ref. [1]
        
    Args:
        a (float): magnitude of off-diagonal elements for lhs and rhs 
        b, c (numpy arrays, ndim=1): field values at previous z-step
        u1, u (numpy arrays, ndim=1): dummy array (will be overwritten) 

    Returns:
        u (numpy array, ndim=1): updated field values at current z-step

    Notes:
        Algorithm is based on the more general procedure discussed in 
        chapter 2.4 Tridiagonal and Band Diagonal Systems of Equations
        of Ref. [2]

        CODi = (C)onstant (O)ff-(Di)agonal elements

    Refs:
        [1] Transparent boundary condition for beam propagation
            G. R. Hadley
            Optics Letters, 16 (1991) 624

        [2] Numerical Recipes in Fortran 77
            Press, WH and Teukolsky, SA and Vetterling, WT and Flannery, BP
            Cambridge University Press (2nd Edition, 2002)

    """
    TINY = 1e-8
    beta = b[0]
    gamma = np.zeros(u.size,dtype=complex)

    r = c[:]*u1[:]
    r[1:] += a*u1[:-1]
    r[:-1] += a*u1[1:]

    u[0]=r[0]/beta
    for j in xrange(1,u.size):
        gamma[j] = -a/beta
        beta = b[j]+a*gamma[j]
        u[j]=(r[j]+a*u[j-1])/beta
    
    for j in xrange(u.size-2,0,-1):
         u[j] = u[j] - gamma[j+1]*u[j+1] 

    return u


def NumRecSolver(a,b,c,r,u):
    """tridiagonal matrix solver 

    Implements tridiagonal matrix solver discussed in chapter 2.4 Tridiagonal
    and Band Diagonal Systems of Equation in Ref. [1]
        
    Args:
        a,b,c,r (numpy arrays, ndim=1): input arrays (not modified) 
        u (numpy arrays, ndim=1): solution vector  

    Returns:
        u (numpy array, ndim=1): solution vector  

    Refs:
        [1] Numerical Recipes in Fortran 77
            Press, WH and Teukolsky, SA and Vetterling, WT and Flannery, BP
            Cambridge University Press (2nd Edition, 2002)

    """
    TINY = 1e-8
    beta = b[0]
    gamma = np.zeros(u.size,dtype=complex)

    if b[1]<TINY:
            sys.stderr.write("Error in TDMSolver")
            exit()

    u[0]=r[0]/beta
    for j in xrange(1,u.size):
        gamma[j] = c[j-1]/beta
        beta = b[j]-a[j]*gamma[j]
        if beta < TINY:
           sys.stderr.write("Error in TDMSolver")
        u[j]=(r[j]-a[j]*u[j-1])/beta
    
    for j in xrange(u.size-2,0,-1):
         u[j] = u[j] - gamma[j+1]*u[j+1] 

    return u


# EOF: tridiagonalMatrixSolver.py  
