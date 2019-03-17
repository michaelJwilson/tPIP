import  matplotlib

import  numpy              as      np
import  pylab              as      pl

from    mcfit              import  xi2P, P2xi
from    scipy.interpolate  import  interp1d
from    smooth             import  bin_smooth


def pip_convert(fname, cumulative=False):
  ##  Given a pip file, get the P(k) multipoles.
  data  = np.loadtxt(fname)

  s     = data[:,0]

  xi0   = data[:,1]
  xi2   = data[:,2]
  xi4   = data[:,3]

  ##  Interpolate on a grid.
  xi0  = interp1d(s, xi0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
  xi2  = interp1d(s, xi2, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
  xi4  = interp1d(s, xi4, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

  ##  New logarithmic r binning.
  rs   = np.logspace(np.log10(0.2), np.log10(165.), 45, endpoint=True, base=10.)

  if cumulative:
    lrs    = np.log10(rs)
    dlr    = np.diff(lrs)[0]

    Ci0    = np.cumsum(xi0(rs) * rs ** 3.) * dlr
    Ci2    = np.cumsum(xi2(rs) * rs ** 3.) * dlr
    Ci4    = np.cumsum(xi4(rs) * rs ** 3.) * dlr 

    ##  And conversion to Fourier.                                                                                                                      
    ks, P0 = xi2P(rs, l=0, lowring=False, deriv=1)(Ci0, extrap=False)
    ks, P2 = xi2P(rs, l=2, lowring=False, deriv=1)(Ci2, extrap=False)
    ks, P4 = xi2P(rs, l=4, lowring=False, deriv=1)(Ci4, extrap=False)

    return  ks, P0, P2, P4

  else:
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
  results['los_one']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStestran_xi_mp.dat', cumulative=cumulative)
  results['los_two']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest05_xi_mp.dat', cumulative=cumulative)
  results['los_three'] = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest2_xi_mp.dat', cumulative=cumulative)
  results['los_four']  = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest_xi_mp.dat', cumulative=cumulative)

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

  ## -- Direct Fourier measurements --
  data    = np.loadtxt('../dat/poles.txt')

  iterate =  2
  pl.loglog(bin_smooth(data[:,0], iterate=iterate), bin_smooth(data[:,0] * data[:,1], iterate=iterate),  'k-', alpha=0.8, label=r'3D FFT $\hat P(k)$')

  pl.loglog(bin_smooth(data[:,0], iterate=iterate), bin_smooth(data[:,0] * data[:,2], iterate=iterate),  'k-', alpha=0.8)

  ##  Plot the truth.
  ks, P0, P2, P4 = results['truth']

  pl.plot(ks,  ks * P0, 'r^', alpha=0.5, markersize=3, label='Hankel(xi)')
  pl.plot(ks,  ks * P2, 'r^', alpha=0.5, markersize=3, label='')

  ##  Mean PIP P0 and error.                                                                                                                                 
  P0s  = np.array(P0s)
  mP0  =  np.mean(P0s, axis=0)
  vP0  =   np.var(P0s, axis=0)

  pl.errorbar(ks, ks * mP0, ks * np.sqrt(vP0), fmt='^', c='c', alpha=0.5, markersize=3, label='PIP')

  ##  Mean PIP P2 and error.                                                                                                                                 
  P2s  = np.array(P2s)
  mP2  = np.mean(P2s, axis=0)
  vP2  =  np.var(P2s, axis=0)

  pl.errorbar(ks, ks * mP2, ks * np.sqrt(vP2), fmt='^', c='m', alpha=0.5, markersize=3, label='')  

  pl.xscale('log')
  pl.yscale('linear')

  pl.xlabel(r'k')
  pl.ylabel(r'$k \cdot P(k)$')

  pl.legend(ncol=1)

  pl.xlim(0.01,    1.)
  ##  pl.ylim(0.,    1.e3)

  pl.show()
  ## pl.savefig('../plots/pk.pdf')
