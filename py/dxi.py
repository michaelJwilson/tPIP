import  matplotlib;  matplotlib.use('Pdf')

import  numpy              as      np
import  pylab              as      pl

from    mcfit              import  xi2P, P2xi
from    scipy.interpolate  import  interp1d
from    scipy.signal       import  medfilt
from    smooth             import  bin_smooth


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

  ##  Plot the truth.
  ks, P0, P2, P4 = results['truth']

  P0 = interp1d(ks, P0, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
  P2 = interp1d(ks, P2, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)

  ##  Direct Fourier measurements.                                                                                                                       
  data = np.loadtxt('../dat/poles.txt')

  iterate = 35
  pl.loglog(bin_smooth(data[:,0], iterate), 100. * (bin_smooth(data[:,1], iterate) / P0(bin_smooth(data[:,0], iterate)) - 1.0),  'k-', alpha=0.8, label=r'3D FFT $P0$')

  iterate = 65
  pl.loglog(bin_smooth(data[:,0], iterate), 100. * (bin_smooth(data[:,2], iterate) / P2(bin_smooth(data[:,0], iterate)) - 1.0),  'r-', alpha=0.8, label=r'3D FFT $P2$')
  
  ##  Mean PIP P0 and error.                                                                                                                                  
  P0s  = np.array(P0s)
  mP0  = np.mean(P0s, axis=0)
  sP0  =  np.std(P0s, axis=0)

  pl.errorbar(ks, 100. * (mP0 / P0(ks) - 1.0), 100. * sP0 / P0(ks), fmt='^', c='c', alpha=0.5, markersize=3, label='PIP')                                              

  ##  Mean PIP P2 and error.                                                                                                                                
  P2s  = np.array(P2s)
  mP2  = np.mean(P2s, axis=0)
  sP2  =  np.std(P2s, axis=0)

  pl.errorbar(ks + 0.005, 100. * (mP2 / P2(ks) - 1.0), 100. * sP2 / P2(ks), fmt='^', c='m', alpha=0.5, markersize=3, label='')

  ##  Error band. 
  ax = pl.gca()
  ax.fill_between(data[:,0], -1., 1., color='b', alpha=0.2)

  pl.xscale('linear')
  pl.yscale('linear')

  pl.xlabel(r'k')
  pl.ylabel(r'$\%$')

  pl.legend(ncol=1)

  pl.xlim(0.1, .5)
  pl.ylim(-5., 5.)

  pl.savefig('../plots/dpk.pdf')
