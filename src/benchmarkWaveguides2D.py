import numpy as np


def squareLawFiberMask((xMin,xMax,Nx)=(-80,80,300), (zMin,zMax,Nz)=(0.,50000.,6000)):

        k0 = 2*np.pi/1.
        nb = 1.5                     
        dn = 0.04665
        a  = 62.5
        alpha = 2.0

        (x, dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
        (z, dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
        xv,zv = np.meshgrid(x, z, indexing='ij')
        nxz = nb*np.ones((Nx,Nz))
        x0 = 0.5*(x[0]+x[-1])

        n1 = nb/np.sqrt(1.-2*dn)

        mask = np.logical_and(xv<a, xv>-a)
        nxz[mask] = n1*np.sqrt(1.-2*dn*(xv[mask]/a)**2)
        
        return (x,dx),(z,dz),k0,nb,nxz 


def straightMask((xMin,xMax,Nx)=(0.,40.,800), (zMin,zMax,Nz)=(0.,100.,1500)):

        k0 = 2*np.pi/1.15
        nb = 2.15                      
        dn = 0.303
        w  = 4.0

        (x, dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
        (z, dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
        xv,zv = np.meshgrid(x, z, indexing='ij')
        nxz = nb*np.ones((Nx,Nz))

        nxz = nb + dn*(np.cosh(2.*(xv-x[x.size/2])/w))**(-2)

        return (x,dx),(z,dz),k0,nb,nxz 


def sBendMask((xMin,xMax,Nx)=(0.,40.,800), (zMin,zMax,Nz)=(0.,40.,800)):

        k0=2.*np.pi/1.55
        nb = 3.3000
        dn = 0.0694
        w = 2.0
        tu = 2.0
        tb = 2.0

        (x, dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
        (z, dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
        xv,zv = np.meshgrid(x, z, indexing='ij')
        nxz = nb*np.ones((Nx,Nz))
        
        x0 = (x[-1]-x[0])/2.0
        nx = lambda zi,t: t*(1.0 - np.cos(np.pi*zi/z[-1])) 

        nxz[np.logical_and(xv < x0-nx(zv,tu)+w/2, 
                        xv > x0-nx(zv,tu)-w/2)] = nb+dn

        return (x,dx),(z,dz),k0,nb,nxz 


def yMask((xMin,xMax,Nx)=(0.,20.,800), (zMin,zMax,Nz)=(0.,800.,1500)):

        k0=2.*np.pi/1.55
        nb = 3.3000
        dn = 0.0694

        w = 2.0
        tu = 2.0
        tb = 2.0

        w = 2.
        tPar = (40.0,600.0,160.0,2.0)
        bPar = (40.0,600.0,160.0,2.0)

        (x, dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
        (z, dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
        xv,zv = np.meshgrid(x, z, indexing='ij')
        nxz = nb*np.ones((Nx,Nz))
        
        x0 = (x[-1]-x[0])/2.0
        xP = lambda zi,(l,b1,d,t):\
                np.piecewise(zi, 
                    [zi<l,
                        np.logical_and(zi>=l, zi<l+b1), 
                        zi>l+b1],
                    [0, 
                        lambda zi: t*(1.0 - np.cos(np.pi*(zi-l)/b1)), 
                        2*t]  
                    )

        nxz[np.logical_and(xv < x0-xP(zv,tPar)+w/2, 
                        xv > x0-xP(zv,tPar)-w/2)] = nb+dn
        nxz[np.logical_and(xv < x0+xP(zv,bPar)+w/2, 
                        xv > x0+xP(zv,bPar)-w/2)] = nb+dn

        return (x,dx),(z,dz),k0,nb,nxz 


def MachZehnderMask((xMin,xMax,Nx)=(0.,30.,1200), (zMin,zMax,Nz)=(0.,2000.,4000)):

        k0=2.*np.pi/1.55
        nb = 3.3
        dn = 0.1

        w = 2.
        tPar = (20.0,650.0,160.0,650.0,520.0,2.0)
        bPar = (20.0,650.0,160.0,650.0,520.0,2.0)

        (x, dx) = np.linspace(xMin,xMax,Nx, endpoint=False, retstep=True)
        (z, dz) = np.linspace(zMin,zMax,Nz, endpoint=False, retstep=True)
        xv,zv = np.meshgrid(x, z, indexing='ij')
        nxz = nb*np.ones((Nx,Nz))
        
        x0 = (x[-1]-x[0])/2.0

        xP = lambda zi,(l,b1,d,b2,r,t):\
                np.piecewise(zi, 
                    [zi<l,
                        np.logical_and(zi>=l, zi<l+b1), 
                        np.logical_and(zi>=l+b1, zi<=l+b1+d),  
                        np.logical_and(zi>=l+b1+d, zi<=l+b1+d+b2),
                        zi>l+b1+d+b2 ],
                    [0, 
                        lambda zi: t*(1.0 - np.cos(np.pi*(zi-l)/b1)), 
                        2*t,  
                        lambda zi: t*(1.0 + np.cos(np.pi*(zi-l-b1-d)/b2)),
                        0]
                    )

        nxz[np.logical_and(xv < x0-xP(zv,tPar)+w/2, 
                        xv > x0-xP(zv,tPar)-w/2)] = nb+dn
        nxz[np.logical_and(xv < x0+xP(zv,bPar)+w/2, 
                        xv > x0+xP(zv,bPar)-w/2)] = nb+dn

        return (x,dx),(z,dz),k0,nb,nxz 


if __name__ == "__main__":
        (x,dx),(z,dz),k0,nb,nxz = MachZehnderMask()
        #(x,dx),(z,dz),k0,nb,nxz = yMask()
        #(x,dx),(z,dz),k0,nb,nxz = straightMask()
        #(x,dx),(z,dz),k0,nb,nxz = sBendMask()
        
# EOF: benchmarkWaveguides2D.py
