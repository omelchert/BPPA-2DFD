import numpy as np

class Domain2D(object):
        def __init__(self,((xMin,xMax),Nx),((zMin,zMax),Nz)):
            (self.x, self.dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
            (self.z, self.dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
            self.grid = np.meshgrid(self.x,self.z, indexing='ij')

        def _idxSet(self,v,(vMin,vMax)):
             return np.logical_and(v>=vMin,v<=vMax)


