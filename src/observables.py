import sys
import numpy as np


class Observables(object):
     def __init__(self,(x,z),(xSamp,zSamp)=(100,100)):
        self.zp = []
        self.LM = []
        self.Eint = []
        self.P1 = []

        self.nx = int(float(x.size)/min(x.size,xSamp))
        self.nz = int(float(z.size)/min(z.size,zSamp))
        self.xIdx = range(0,x.size,self.nx) 
        self.zIdx = range(0,z.size,self.nz) 
        self.x = x[self.xIdx]
        self.z = z[self.zIdx]

        self.FProf = [] 

     def measure(self,n,(x,dx),(z,dz),E0,E):
         """perform measurement 

         Implements callback function for use with paraxial wave equation solver
         that performs measurement on the data provided at a given iteration
         step (i.e. propagation depth)

         Args:
             n (int): current iteration step
             (x,dx) (numpy array; float): x-grid and corresponding mesh width dx 
             (z,dz) (numpy array; float): z-grid and corresponding mesh width dz
             E1 (numpy array, ndim=1): previous field profile
             E (numpy array, ndim=1): current field profile

         Note:
             used as callback function, i.e. argument number 5, in the 
             paraxial wave equation solver implemented in module
             finiteDifferenceBeamPropagation2D.py
         """
         self.zp.append(n*dz)
         self._modeMismatchLoss(dx,E0,E)
         self._correlationFunction(dx,E0,E)
         self._fieldProfile(n,E)
         self._fieldIntensity(dx,E)

     def _modeMismatchLoss(self,dx,E0,E):
        """mode mismatch loss 

        Implements mode mismatch loss for 2D waveguide according to Ref. [1]

        Args:
            dx (float): meshwidth in x direction
            E0 (numpy array, ndim=1): incident field of the fundamental mode
            E (numpy array, ndim=1): propagating field

        Refs:
            [1] Modified Finite-Difference Beam Propagation Method Based on the
                Generalized Douglas Scheme for Variable Coefficients
                Yamauchi, J. and Shibayama, J. and Nakano, H.
                IEEE Photonics Tech. Lett., 7 (1995) 661

        """
        A = np.abs(np.trapz(E0*np.conj(E),dx=dx))**2
        B = np.abs(np.trapz(np.abs(E0)**2,dx=dx))**2
        self.LM.append(-10.0*np.log10(A/B))

     def _correlationFunction(self,dx,E0,E):
        """correlation function

        Implements complex field-amplitude correlation function following [1]

        Args:
            dx (float): meshwidth in x direction
            E0 (numpy array, ndim=1): incident field of the fundamental mode
            E (numpy array, ndim=1): propagating field

        Refs:
            [1] Computation of mode properties in optical fiber waveguides
                by a propagating beam method
                Feit, M. D. and Flec, J. A. Jr.
                Applied Optics, 19 (1980) 1154

        """
        self.P1.append(np.trapz(E0*np.conj(E),dx=dx))

     def _powerAttenuation(self,P):
         """power attenuation
        
         Implements power attenuation following Ref. [1]

         Args:
             P (numpy array, ndim=1): field intensity array

         Note:
             for lossless straight waveguide, the power attenuation is due
             solely to numerical dissipation 

         Refs:
             [1] Modified finite-difference beam-propagation method based on the 
                 Douglas scheme
                 Sun, L. and Yip, G. L.
                 Optics Letters, 18 (1993) 1229

         """
         return [-10.*np.log10(P[i]/P[0]) for i in range(len(P))]

     def _fieldIntensity(self,dx,E):
        """field intensity 

        Implements field intensity Eint = \int_x E E* dx at given z

        Args:
            dx (float): meshwidth in x direction
            E (numpy array, ndim=1): propagating field
        """
        self.Eint.append(np.trapz(np.abs(E)**2,dx=dx))

     def _fieldProfile(self,n,E):
         """accumulate discretized field profile"""
         if n%self.nz == 0:
             self.FProf.append(E[self.xIdx])


     def dumpField(self,func=np.abs,fName=None):
         """dump field 

         Args:
             func (function): operation to perform on complex field value
                 prior to writing it to outstream
             fName (str): filename (including path) to which ascii output
                 should be written

         Note:
             if no filename is provided, ascii output is written to 
             standard outstream
         """
         fStream = sys.stdout if fName == None else open(fName,'w')

         fStream.write("%d "%(self.x.size))
         for i in range(self.x.size):
             fStream.write("%lf "%(self.x[i]))
         fStream.write("\n")

         for j in range(self.z.size):
             fStream.write("%lf "%(self.z[j]))
             for i in range(self.x.size):
                 fStream.write("%lf "%(func(self.FProf[j][i])))
             fStream.write("\n")

     def dumpObservables(self,fName=None):
         """dump field 

         Args:
             func (function): operation to perform on complex field value
                 prior to writing it to outstream
             fName (str): filename (including path) to which ascii output
                 should be written

         Note:
             if no filename is provided, ascii output is written to 
             standard outstream
         """
         fStream = sys.stdout if fName == None else open(fName,'w')

         alpha = self._powerAttenuation(self.Eint)
         fStream.write("# z LM (E,E) abs(P1) alpha\n")
         for i in range(len(self.zp)):
             fStream.write("%lf %lf %lf %lf %lf\n"%(
                 self.zp[i], self.LM[i], self.Eint[i], np.abs(self.P1[i]), 
                 alpha[i]))


# EOF: observables.py       
