import numpy as np


def TBC2D(dx,(E11,E12),(E01,E02)):
    """impose transparent boundary conditions

    Implements simple transparent boundary condition for 2D beam propagation as
    discussed by Hadley in Ref. [1]
    
    Args:
        dx (float): increment along x-axis  
        (E11, E12) (float tuple): consequtive field values at previous z-step 
        (E01, E02) (float tuple): consequtive field values at current z-step 

    Returns:
        E02 (float): possibly modified field value 

    Notes:
        Since we use the field distribution dependence considered by Ref. [2],
        i.e. E(x,z+dz) = E(x,z) \exp(-i kx dx), which differns by the minus
        sign in the exponent from the definition used by Ref. [1], our updating  
        formula is modified accordingly.

    Refs:
        [1] Transparent boundary condition for beam propagation
            G. R. Hadley
            Optics Letters, 16 (1991) 624

        [2] An Assessment of Finite Difference Beam Propagation Method
            Chung, Y. and Dagli, N.
            IEEE J. Quant. Elect., 26 (1990) 1335

    """
    if np.abs(E11) > 1e-6:
        # compute kx previous to update following Eq. (4) of Ref. [1]
        kx = 1j*np.log(E11/E12)/dx
        # restrict real part of kx as discussed by Ref. [1] after Eq. (5) 
        kx = kx if kx.real > 0. else 1j*kx.imag
        # update new boundary field value following Eq. (5) of Ref. [1]
        E02 = E01*np.exp(-1j*kx*dx)
    return E02


# EOF: boundaryConditions.py
