import  matplotlib

import  numpy              as      np
import  pylab              as      pl
import  mcfit

from    mcfit              import  xi2P, P2xi, kernels
from    scipy.interpolate  import  interp1d
from    smooth             import  bin_smooth


def pip_convert(fname, cumulative=False):
  ##  Given a pip file, get the P(k) multipoles.
  data  = np.loadtxt(fname)

  s     = data[:,0]

  xi0   = data[:,1]
  xi2   = data[:,2]
  xi4   = data[:,3]

  if cumulative:
    ##  ss  = np.logspace(np.log(5.), np.log(150.), num=100, base=np.exp(1))
    ss      = numpy.logspace(-3, 3, num=60, endpoint=False)
    A       = 1 / (1 + x*x)**1.5

    '''
    steps   = np.diff(np.log(ss))
    step    = steps[0]

    ##  Interpolate on a grid.                                                                                                                                                                                                                                    
    Ci0     = interp1d(s, s ** 3. * xi0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
    Ci2     = interp1d(s, s ** 3. * xi2, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
    Ci4     = interp1d(s, s ** 3. * xi4, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

    Ci0     = np.sum(Ci0(ss)) * step
    Ci2     = np.sum(Ci2(ss)) * step
    Ci4     = np.sum(Ci4(ss)) * step

    result  = []

    for nu, phase, CC in zip([0, 2, 4], [1., -1., 1.], [Ci0, Ci2, Ci4]):
      print('Solving for nu: %d' % nu)

      H1      =  mcfit.mcfit(ss, mcfit.kernels.Mellin_SphericalBesselJ(nu - 1, deriv=0), q=0)
      H2      =  mcfit.mcfit(ss, mcfit.kernels.Mellin_SphericalBesselJ(nu,     deriv=0), q=0)

      y1, B1  =  H1(ss * CC)
      y2, B2  =  H2(ss * CC / ss)

      Pl      =  4. * np.pi * phase * (-B1 + (nu + 1.0) * B2 / y2)  

      result.append(Pl) 
      '''

    return  y1, result[0], result[1], result[2]

  else:
    ##  New logarithmic r binning.                                                                                                                                                                                                                                 
    rs   = np.logspace(np.log10(0.2), np.log10(165.), 45, endpoint=True, base=10.)

    ##  Interpolate on a grid.                                                                                                                                                                                                                                      
    xi0  = interp1d(s, xi0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
    xi2  = interp1d(s, xi2, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
    xi4  = interp1d(s, xi4, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

    xi0  = xi0(rs)
    xi2  = xi2(rs)
    xi4  = xi4(rs)
    
    ##  And conversion to Fourier.
    ks, P0 = xi2P(rs, l=0, lowring=False, deriv=0)(xi0, extrap=False)
    ks, P2 = xi2P(rs, l=2, lowring=False, deriv=0)(xi2, extrap=False)
    ks, P4 = xi2P(rs, l=4, lowring=False, deriv=0)(xi4, extrap=False)
    
    return  ks, P0, P2, P4


if __name__ == '__main__':
  cumulative           = False
  results              = {}

  results['truth']     = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_rls5nb45_xi_mp.dat', cumulative=cumulative)

  '''
  results['los_one']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStestran_xi_mp.dat', cumulative=cumulative)
  results['los_two']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest05_xi_mp.dat', cumulative=cumulative)
  results['los_three'] = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest2_xi_mp.dat', cumulative=cumulative)
  results['los_four']  = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest_xi_mp.dat', cumulative=cumulative)
  '''
  '''
  for dd in [1, 2, 3, 4]:
    fname                  = '../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_s%d_PIP992tm6nb21rls5nb45_p3_xi_mp.dat' % dd
    results['PIP_%d' % dd] = pip_convert(fname, cumulative=cumulative)

  P0s = []
  P2s = []
  P4s = []

  for key in ['PIP_1', 'PIP_2', 'PIP_3', 'PIP_4']:
    ks, P0, P2, P4 = results[key]

    P0s.append(P0)
    P2s.append(P2)
    P4s.append(P4)
  '''

  ## -- Direct Fourier measurements --
  data    = np.loadtxt('../dat/poles.txt')

  ##  Box car smoothing of P(k).
  iterate =  2

  pl.loglog(bin_smooth(data[:,0], iterate=iterate), bin_smooth(data[:,0] * data[:,1], iterate=iterate),  'k-', alpha=0.8, label=r'3D FFT $\hat P(k)$')
  pl.loglog(bin_smooth(data[:,0], iterate=iterate), bin_smooth(data[:,0] * data[:,2], iterate=iterate),  'k-', alpha=0.8)

  ##  Plot the Hankel transforms of the measured correlation fn.  
  ks, P0, P2, P4 = results['truth']

  pl.plot(ks,  ks * P0, 'r^', alpha=0.5, markersize=3, label='Hankel(xi)')
  pl.plot(ks,  ks * P2, 'r^', alpha=0.5, markersize=3, label='')

  '''
  ##  Plot the mean PIP-corrected P0 and error.                                                                                                                                 
  P0s  = np.array(P0s)
  mP0  =  np.mean(P0s, axis=0)
  vP0  =   np.var(P0s, axis=0)

  pl.errorbar(ks, ks * mP0, ks * np.sqrt(vP0), fmt='^', c='c', alpha=0.5, markersize=3, label='PIP')

  ##  Plot the mean PIP-corrected P2 and error.                                                                                                                                 
  P2s  = np.array(P2s)
  mP2  = np.mean(P2s, axis=0)
  vP2  =  np.var(P2s, axis=0)

  pl.errorbar(ks, ks * mP2, ks * np.sqrt(vP2), fmt='^', c='m', alpha=0.5, markersize=3, label='')  
  '''
  pl.xscale('log')
  pl.yscale('linear')

  pl.xlabel(r'k')
  pl.ylabel(r'$k \cdot P(k)$')

  pl.legend(ncol=1)

  pl.xlim(0.01,    1.)
  ##  pl.ylim(0.,    1.e3)

  pl.show()
  ## pl.savefig('../plots/pk.pdf')
