import  matplotlib;  matplotlib.use('Pdf')

import  numpy              as      np
import  pylab              as      pl

from    mcfit              import  xi2P, P2xi
from    scipy.interpolate  import  interp1d
from    scipy.signal       import  medfilt


def pip_convert(fname):
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

  xi0  = xi0(rs)
  xi2  = xi2(rs)
  xi4  = xi4(rs)

  ##  And conversion to Fourier.
  ks, P0 = xi2P(rs, l=0, lowring=False)(xi0, extrap=False)
  ks, P2 = xi2P(rs, l=2, lowring=False)(xi2, extrap=False)
  ks, P4 = xi2P(rs, l=4, lowring=False)(xi4, extrap=False)

  return  ks, P0, P2, P4


if __name__ == '__main__':
  results = {}

  results['truth']     = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_rls5nb45_xi_mp.dat')

  results['los_one']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStestran_xi_mp.dat')
  results['los_two']   = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest05_xi_mp.dat')
  results['los_three'] = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest2_xi_mp.dat')
  results['los_four']  = pip_convert('../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_t_LOStest_xi_mp.dat')

  for dd in [1, 2, 3, 4]:
    fname                  = '../dat/DESI_ra130-230_dec20-60_z0-3_1_h100_s%d_PIP992tm6nb21rls5nb45_p3_xi_mp.dat' % dd
    results['PIP_%d' % dd] = pip_convert(fname)

  P0s = []
  P2s = []
  P4s = []

  for key in ['PIP_1', 'PIP_2', 'PIP_3', 'PIP_4']:
    ks, P0, P2, P4 = results[key]

    P0s.append(P0)
    P2s.append(P2)
    P4s.append(P4)

    ##  pl.loglog(ks, P0, 'm', alpha=0.1, markersize=3)
    ##  pl.loglog(ks, P2, 'm', alpha=0.1, markersize=3)

  ##  Mean PIP P0 and error.
  P0s  = np.array(P0s)
  mP0  = np.mean(P0s, axis=0)
  vP0  =  np.var(P0s, axis=0)

  ##  pl.errorbar(ks, mP0, np.sqrt(vP0), fmt='^', c='c', alpha=0.5, markersize=3, label='Assignments -- Hankel')

  ##  Mean PIP P2 and error.
  P2s  = np.array(P2s)
  mP2  = np.mean(P2s, axis=0)
  vP2  =  np.var(P2s, axis=0)

  ##  pl.errorbar(ks, mP2, np.sqrt(vP2), fmt='^', c='c', alpha=0.5, markersize=3, label='')

  '''
  ## -- Direct Fourier measurements --
  data = np.loadtxt('../dat/_poles.txt')

  pl.loglog(data[:,0], data[:,1],  'y-', alpha=0.3, label='Data -- Direct FFT (Old)')
  pl.loglog(data[:,0], data[:,2],  'y-', alpha=0.3)
  '''

  ##  Plot the truth.
  ks, P0, P2, P4 = results['truth']

  P0 = interp1d(ks, P0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
  P2 = interp1d(ks, P2, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

  ## -- Direct Fourier measurements --                                                                                                                       
  data = np.loadtxt('../dat/poles.txt')

  pl.loglog(data[:,0], 100. * (medfilt(data[:,1], 101) / P0(data[:,0]) - 1.0),  'k-', alpha=0.8, label=r'$P0$')
  pl.loglog(data[:,0], 100. * (medfilt(data[:,2], 101) / P2(data[:,0]) - 1.0),  'r-', alpha=0.8, label=r'$P2$')

  '''
  ## Plot los versions.
  ks, P0, P2, P4 = results['los_one']

  pl.plot(ks,  P0, 'k^', alpha=0.5, markersize=3, label='LOS -- Hankel')
  pl.plot(ks,  P2, 'k^', alpha=0.5, markersize=3, label='')
  '''

  pl.xscale('linear')
  pl.yscale('linear')

  pl.xlabel(r'k')
  pl.ylabel(r'$\Delta P(k)$')

  pl.legend(ncol=1)

  pl.xlim(0.1,      .5)
  pl.ylim(-1.e1,  1.e1)

  pl.savefig('../plots/dpk.pdf')
