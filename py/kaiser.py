##############                                                                                                                                                                                                                 
P    = interp1d(k, P,  kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)
ks   = np.logspace(-3., -0.5, 150, endpoint=True, base=10.)

P    = P(ks)

beta = 0.3

P0   = (1. + 2./3. * beta + beta * beta / 5.) * P
P2   = (4. * beta / 3. + 4. * beta * beta / 7.) * P
P4   =  8. * beta * beta * P /35

rs, X0   = P2xi(ks, l=0)(P0)
rs, X2   = P2xi(ks, l=2)(P2)
rs, X4   = P2xi(ks, l=4)(P4)

XQ0      = X0 * Q0(rs) + X2 * Q2(rs) / 5. + X4 * Q4(rs) / 9.
XQ2      = X0 * Q2(rs) + X2 * (Q0(rs) + 2. * Q2(rs) / 7. + 2. * Q4(rs) / 7.)

ks, P0   = xi2P(rs, l=0)(XQ0)
ks, P2   = xi2P(rs, l=2)(XQ2)

#pl.plot(ks, P0, 'm-', label='Linear * Q0', alpha=0.5)                                                                                                                                                                         
#pl.plot(ks, P2, 'm-', label='Linear * Q2', alpha=0.5)                                                                                                                                                                         

## IC                                                                                                                                                                                                                          
ks, W0   = xi2P(rs, l=0)(Q0(rs))
ks, W2   = xi2P(rs, l=2)(Q2(rs))

A        = 1.e-2

#pl.plot(ks, P0 - A * W0, 'm-', label='Linear * Q0', alpha=1.0)                                                                                                                                                                
#pl.plot(ks, P2 - A * W2, 'm-', label='Linear * Q2', alpha=0.5)
