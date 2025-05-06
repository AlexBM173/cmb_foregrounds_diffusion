import numpy as np
import torch
import healpy as hp
import astropy.units as u
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


@u.quantity_input
def get_patch_centers(gal_cut: u.deg, step_size: u.deg):
    """ Function to get the centers of the various patches to be cut out.

    Parameters
    ----------
    gal_cut: float
        We will miss out the region +/- `gal_cut` in Galactic latitude, measured
        in degrees.
    step_size: float
        Stepping distance in Galactic longitude, measured in degrees, between 
        patches.

    Returns
    -------
    list(tuple(float))
        List of two-element tuples containing the longitude and latitude.
    """
    gal_cut = gal_cut.to(u.deg)
    step_size = step_size.to(u.deg)
    assert gal_cut.unit == u.deg
    assert step_size.unit == u.deg
    southern_lat_range = np.arange(-90, (-gal_cut-step_size).value, step_size.value) * u.deg
    northern_lat_range = np.arange((gal_cut + step_size).value, 90, step_size.value) * u.deg
    lat_range = np.concatenate((southern_lat_range, northern_lat_range))

    centers = []
    for t in lat_range:
        step = step_size.value / np.cos(t.to(u.rad).value)
        for i in np.arange(0, 360, step):
            centers.append((i * u.deg, t))
    return centers

class FlatCutter(object):
    """ Object to control the extraction of flat patches from a given HEALPix 
    map.

    Object is initialized with parameters defining the geometry of the patch:
    its length in degrees, and the number of pixels in each direction. 
    The `rotate_and_interpolate` method defines a grid centered at (0, 0)
    of dimensions corresponding to `xlen`, `ylen`, and rotates it to the 
    point (lon, lat). The value of the map at the resulting grid of longitudes 
    and latitudes is then determined by interpolation. 
    """
    @u.quantity_input
    def __init__(self, ang_x: u.deg, ang_y: u.deg, xres, yres):
        assert type(xres) is int
        assert type(yres) is int
        self.xres = xres
        self.yres = yres

        self.ang_x = ang_x
        self.ang_y = ang_y
        
        # get grid of unit vectors corresponding to flat patch around
        # pole (z = 1). For this we use ang_x in radians, as is appropriate
        # for the implicit small-angle approximation 
        self.xarr = np.linspace(- self.ang_x.to(u.rad).value / 2., 
                                self.ang_x.to(u.rad).value / 2., xres)
        self.yarr = np.linspace(- self.ang_y.to(u.rad).value / 2., 
                                self.ang_y.to(u.rad).value / 2., yres)

        xgrid, ygrid = np.meshgrid(self.xarr, self.yarr)
        xgrid = xgrid.ravel()[None, :]
        ygrid = ygrid.ravel()[None, :]
        zgrid = np.ones_like(ygrid)
        
        # vectors corresponding to cartesian grid around poll
        self.vecs = np.concatenate((xgrid, ygrid, zgrid)).T

        # get the latitude (*not colatitude*) and longitude in degrees
        # of the cartesian grid points around the pole. 
        self.lons, self.lats = hp.vec2ang(self.vecs, lonlat=True)
        self.lats *= u.deg
        self.lons *= u.deg
        return
    
    @u.quantity_input
    def rotate_to_pole_and_interpolate(self, lon: u.deg, lat: u.deg, ma):
        """ Method to rotate the grid at (0, 0) to `rot=(lon, lat)`, and sample
        the map at the grid points by interpolation.

        Parameters
        ----------
        lat, lon: float
            Latitude (*not* colatitude) and longitude of point to be rotated
            to the North pole, in degrees.
        ma: ndarray
            Healpix map from which the interpolation is to be made.
        """
        if hp.pixelfunc.maptype(ma) == 0:  # a single map is converted to a list
            ma = [ma]
        # define a rotation object in terms of the theta_rot and phi_rot angles.
        # This returns a rotator object that can be applied to rotate a given
        # vector by this angle. Since we are interested in rotating some patch
        # to the pole, we actually want to apply the *inverse* rotation operator
        # to the vectors self.co_lats, self.lons.
        lon = lon.to(u.deg)
        lat = lat.to(u.deg)
        rotator = hp.Rotator(rot=[lon.value, lat.value - 90.], deg=True)
        self.inv_lon_grid, self.inv_lat_grid = rotator.I(self.lons.to(u.deg).value, self.lats.to(u.deg).value, lonlat=True)
        # Interpolate the original map to the pixels centers in the new ref frame
        m_rot = [hp.get_interp_val(each, self.inv_lon_grid, self.inv_lat_grid, lonlat=True) for each in ma]

        # Rotate polarization
        if len(m_rot) > 1:
            # Create a complex map from QU  and apply the rotation in psi due to the rotation
            # Slice from the end of the array so that it works both for QU and IQU
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], rotator.angle_ref(self.inv_lon_grid, self.inv_lat_grid, lonlat=True))
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], self.lons.to(u.rad).value)
        else:
            m_rot = m_rot[0]
        return np.moveaxis(np.array(m_rot).reshape(-1, self.xres, self.yres), 0, -1)
    

def replace_zeros_with_neighbor_avg(hp_map):
    """
    Replace zero values in a HEALPix map with the average of their non-zero neighbors.

    This function scans through the provided HEALPix map and replaces each zero-value pixel
    with the average value of its neighboring pixels that are non-zero. If no valid non-zero
    neighbors are found, the zero value may be replaced with a predefined value or left unchanged,
    depending on the desired behavior for such cases.

    Parameters:
    hp_map (np.ndarray): A 1D numpy array representing a HEALPix map with NSIDE resolution.

    Returns:
    np.ndarray: The modified HEALPix map with zero values replaced by the average of their non-zero neighbors.
    """
    nside = hp.get_nside(hp_map)
    zeros_indices = np.where(hp_map == 0)[0]
    
    for idx in zeros_indices:
        neighbors = hp.get_all_neighbours(nside, idx)
        # Filter out neighbors with invalid indices (-1) and zero values in the map
        valid_neighbors = neighbors[(neighbors >= 0) & (hp_map[neighbors] != 0)]
        
        if len(valid_neighbors) > 0:
            # Calculate the average of valid, non-zero neighbors
            average_value = np.mean(hp_map[valid_neighbors])
            hp_map[idx] = average_value
        else:
            # If no valid neighbors are found, handle this edge case appropriately
            # Here, you might set it to a default value or leave it unchanged
            hp_map[idx] = 0 # Or any other appropriate value
            
    return hp_map

def split_data_to_tensors(data, train_size=0.7, val_size=0.15, test_size=0.15,seed=42):
    """
    Splits the data into training, validation, and test datasets based on provided sizes,
    rearranges the data for PyTorch, and converts them into tensors.
    
    Parameters:
        data (np.array): The input data with shape (b, h, w, c).
        train_size (float): Fraction of the data to be used for training.
        val_size (float): Fraction of the data to be used for validation.
        test_size (float): Fraction of the data to be used for testing.
    
    Returns:
        tuple: A tuple containing training, validation, and test datasets as PyTorch tensors.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("The sum of train_size, val_size, and test_size should be 1.")
    rng = np.random.default_rng(seed)
    
    # Shuffling indices
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)

    # Calculate split indices
    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    # Splitting the data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Rearrange data to match PyTorch's (batch, channels, height, width) format
    train_set = torch.tensor(data[train_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    val_set = torch.tensor(data[val_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    test_set = torch.tensor(data[test_indices].transpose(0, 3, 1, 2), dtype=torch.float32)

    return train_set, val_set, test_set

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d
def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    nx, ny, dx, dy = flatskymapparams
    dx_rad = np.radians(dx/60.)
    dy_rad = np.radians(dy/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
            flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    return el, cl

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dy = flatskymapparams
    dx = np.radians(dx/60.)
    dy = np.radians(dy/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dy ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(z.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval

    return radprf