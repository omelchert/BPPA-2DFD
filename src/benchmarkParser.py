import scipy.io


def parseInstance(fName):

    matDict = scipy.io.loadmat(fName)

    (x,dx) = (matDict['x'][0], matDict['dx'][0,0])
    (z,dz) = (matDict['z'][0], matDict['dz'][0,0])
    k0 = matDict['k0'][0,0]
    nxz = matDict['nxz']
    nEff = matDict['nEff'][0,0]
    E0 = matDict['E0'][0]

    return (x,dx),(z,dz), k0, nxz, nEff, E0


# EOF: benchmarkParser.py
