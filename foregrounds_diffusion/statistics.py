import numpy as np
from scipy import optimize

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    xgrid, ygrid = np.indices(data.shape)
    x = (xgrid*data).sum()/total
    y = (ygrid*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitting_func(p, p0, xgrid, ygrid, tmap, lbounds = None, ubounds = None, fixed = None, return_fit = 0):
    if hasattr(fixed, '__len__'):
        p[fixed] = p0[fixed]

    if hasattr(lbounds, '__len__'):
        linds = abs(p)<abs(lbounds)
        if len(linds)>0: return tmap
        p[linds] = lbounds[linds]

    if hasattr(ubounds, '__len__'):
        uinds = abs(p)>abs(ubounds)
        if len(uinds)>0: return tmap
        p[uinds] = ubounds[uinds]

    def gaussian(p, xp, yp):
        if len(p)>6:
            wx, wy = p[4], p[5]
            rota = np.radians(p[6])
            xp = np.radians(xp/60.) * np.cos(rota) - np.radians(yp/60.) * np.sin(rota)
            yp = np.radians(xp/60.) * np.sin(rota) + np.radians(yp/60.) * np.cos(rota)

            xp = np.degrees(xp) * 60.
            yp = np.degrees(yp) * 60.
        else:
            wx = wy = p[4]
        ##print(p); sys.exit()
        g = p[0]+p[1]*np.exp( -(((p[2]-xp)/wx)**2+ ((p[3]-yp)/wy)**2)/2.)

        return g

    if not return_fit:
        return np.ravel(gaussian(p, xgrid, ygrid) - tmap)
    else:
        return gaussian(p, xgrid, ygrid)
    
def apply_stdnorm(maps):
    """
    Apply standard normalization per channel: (x - mean) / std, channel-wise.

    Parameters:
    - maps: np.ndarray, shape (..., C) where C is the number of channels

    Returns:
    - np.ndarray with same shape, normalized per channel
    """
    maps = maps.copy()
    for c in range(maps.shape[-1]):
        channel = maps[..., c]
        maps[..., c] = (channel - np.mean(channel)) / np.std(channel)
    return maps
    
def stats(maps):
    return np.min(maps), np.max(maps), np.mean(maps), np.std(maps)
    
def renormalize_dm_maps(dm_maps, train_maps, variance_scaling=True):
    """
    Renormalizes diffusion model maps to match the mean and variance of training maps.

    Parameters:
    - dm_maps: np.ndarray, shape (N, C, H, W)
    - train_maps: np.ndarray, shape (N, H, W, C)
    - variance_scaling: bool, if True, match variance in addition to mean

    Returns:
    - np.ndarray of shape (N, C, H, W), renormalized
    """
    dm_maps = np.transpose(dm_maps, (0, 2, 3, 1)).copy()
    num_channels = train_maps.shape[-1]

    for i in range(num_channels):
        tr_min = np.min(train_maps[:, :, :, i])
        tr_max = np.max(train_maps[:, :, :, i])

        dm_maps[:,:,:, i] = dm_maps[:,:,:, i] * (tr_max - tr_min) + tr_min

        if variance_scaling:
            dm_mean = np.mean(dm_maps[:,:,:, i])
            dm_std = np.std(dm_maps[:,:,:, i])
            tr_mean = np.mean(train_maps[:, :, :, i])
            tr_std = np.std(train_maps[:, :, :, i])
            dm_maps[:,:,:, i] = (dm_maps[:,:,:, i] - dm_mean) * (tr_std / dm_std) + tr_mean

    return dm_maps