import  numpy              as      np
import  pylab              as      pl

from    mcfit              import  xi2P, P2xi
from    scipy.interpolate  import  interp1d
from    xi                 import  pip_convert


if __name__ == '__main__':
  ## Load linear P(k), lin_pmm.dat
  data    = np.loadtxt('../dat/lin_pmm.dat')

  ks      = data[:,0]
  P0      = data[:,1]

  P0      = interp1d(ks, P0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

  ## New logarithmic k binning.                                                                                                             
  ks      = np.logspace(-3.0, np.log10(3.), 45, endpoint=True, base=10.)
  P0      = P0(ks)

  pl.loglog(ks, P0, 'k-')
  
  rs, x2  = P2xi(ks, l=2)(P0)
  ks, P2  = xi2P(rs, l=2)(x2)

  pl.loglog(ks, P2, 'k^', markersize=3)
             
  ## Noise realisations.     
  result = []
     
  upper  = 25

  print  rs[upper]

  for i in np.arange(3000):
    noise           = np.copy(x2)
    noise[3:upper] += np.random.uniform(0.0, .2 * np.abs(x2[3:upper]), len(x2[3:upper]))

    ks, PN          = xi2P(rs, l=2)(noise)
  
    result.append(PN)

  ### 
  result    = np.array(result)

  mean      = np.mean(result, axis=0)
  var       =  np.var(result, axis=0)
  
  pl.errorbar(ks, np.abs(mean), np.sqrt(var), fmt='m')

  pl.xscale('log')
  pl.yscale('log')

  ## pl.xlabel(r'k')

  pl.legend(ncol=1)

  pl.xlim(1.e-2,    1.)
  pl.ylim(1.e+1,  4.e4)

  pl.savefig('../plots/noise.pdf')
