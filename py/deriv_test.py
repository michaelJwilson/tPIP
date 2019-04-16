import  mcfit
import  numpy  as  np
import  pylab  as  pl


x      =  np.logspace(-5, 5, num=1e4, endpoint=False)
A      =  1. / (1. + x * x) ** 1.5

step   =  np.diff(np.log(x))[0]
G      =  np.cumsum(A * x ** 2.) * step 

##  Explicit deriv.
nu     =  0

H1     =  mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(-1, deriv=0), q=0, lowring=True)
y1, B1 =  H1(x * G) 

H2     =  mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ( 0, deriv=0), q=0)
y2, B2 =  H2(G)

result = - B1 * y1  ##  + nu * B2
pl.loglog(x, result, 'c-')


##  MCFIT deriv. 
H1     =  mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(0, deriv=1), q=0, lowring=True)
y1, B1 =  H1(x * G)

pl.loglog(y1, -y1 * B1, 'g-')

##                                                                                                                                                                                                                                                                  
H      =  mcfit.mcfit(x, mcfit.kernels.Mellin_BesselJ(0, deriv=0), q=1)
y, B   =  H(x**2 * A)

pl.loglog(x, np.exp(-y), 'k-')
pl.loglog(x, B, 'r--')

pl.xlim(1.e-3, 30.)
pl.ylim(1.e-8, 1.3)
pl.show()
