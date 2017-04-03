import sys; sys.path.append('../../../src/')
import numpy as np
from finiteDifferenceBeamPropagation2D import *
from benchmarkWaveguides2D import *
from observables import *
from guidingModes import *


def GaussianBeam(x,x0,s):
    """normalized Gaussian beam profile
    
    Args:
        x (numpy array, ndim=1): discretized x domain
        x0 (float): x-offset of beam
        s (float): width of beam (s*s is beam variance)
    
    Notes:
        Upon output, beam profile is normalized so that \int_x E E* dx = 1

    Returns:
        f (numpy array, ndim=1): normalized beam profile
    """
    E = np.exp(-(x-x0)**2/2/s/s)
    norm = np.sqrt(np.trapz(np.abs(E)**2,dx=x[1]-x[0]))
    return E/norm


def main():
    cmd = sys.argv
    # GENERATE WAVEGUIDE INSTANCE #############################################
    xMin, xMax, Nx = -80., 80., 400                        # specify x-mesh 
    zMin, zMax, Nz = 0., 2000., 2000                       # specigy z-mesh
    xOff = float(cmd[1])                                   # input beam offset      
    fNameObs = './data/observables_xOff%04.2lf.dat'%(xOff) # ascii output obs
    fNameField = './data/field_xOff%04.2lf.prof'%(xOff)    # ascii output field

    # SET CUSTOM REFRACTIVE INDEX PROFILE ##################################### 
    args = (xMin,xMax,Nx), (zMin,zMax,Nz)
    (x,dx),(z,dz),k0,nb,nxz = squareLawFiberMask(*args)

    # DETERMINE GUIDING MODES THROUGH MODE OPERATOR ANALYSIS  #################
    TEmodes = boundStates((x,dx),k0,nxz[:,0],nb)
    nEff, E0 = TEmodes[0]

    # GENERATE INSTANCE OF OBSERVABLE CLASS AND SET FIELD MONITOR RESOLUTION ## 
    obs = Observables((x,z),(100,100))

    # SET GAUSSIAN INITIAL BEAM ###############################################
    E = GaussianBeam(x, 0.5*(x[0]+x[-1]) + xOff, 10.24) 

    # PROPAGATE BEAM THROUGH COMPUTATIONAL DOMAIN #############################
    Yamauchi1996Solver(((x,dx),(z,dz)),(nxz,nEff,k0),E,obs.measure)

    # SHOW FIELD PROFILE ######################################################
    obs.dumpObservables(fNameObs)
    obs.dumpField(np.abs,fNameField)


main()
# EOF: main_solveInstance.py
