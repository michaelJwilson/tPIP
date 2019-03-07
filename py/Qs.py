import  numpy              as      np
import  pylab              as      pl

from    scipy.interpolate  import  interp1d


data = np.loadtxt('../dat/Ql.dat')

Q0   = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=True, assume_sorted=False)
Q2   = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=True, assume_sorted=False)
Q4   = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=True, assume_sorted=False)
