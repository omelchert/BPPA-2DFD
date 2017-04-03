import sys; sys.path.append('../../../src/')
import numpy as np
from finiteDifferenceBeamPropagation2D import *
from benchmarkWaveguides2D import *
from observables import *
from guidingModes import *


def main():
    cmd = sys.argv
    # GENERATE WAVEGUIDE INSTANCE #############################################
    fNameObs = './data/observables_guidingMode.dat'        # ascii output obs
    fNameField = './data/field_guidingMode.prof'           # ascii output field

    # SET CUSTOM REFRACTIVE INDEX PROFILE ##################################### 
    (x,dx),(z,dz),k0,nb,nxz = straightMask()

    # DETERMINE GUIDING MODES THROUGH MODE OPERATOR ANALYSIS  #################
    TEmodes = boundStates((x,dx),k0,nxz[:,0],nb)
    nEff, E0 = TEmodes[0]

    # GENERATE INSTANCE OF OBSERVABLE CLASS AND SET FIELD MONITOR RESOLUTION ## 
    obs = Observables((x,z),(100,100))

    # PROPAGATE BEAM THROUGH COMPUTATIONAL DOMAIN #############################
    Yamauchi1996Solver(((x,dx),(z,dz)),(nxz,nEff,k0),E0,obs.measure)

    # SHOW FIELD PROFILE ######################################################
    obs.dumpObservables(fNameObs)
    obs.dumpField(np.abs,fNameField)


main()
# EOF: main_solveInstance_guidingMode.py
