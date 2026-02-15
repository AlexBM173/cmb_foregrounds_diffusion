import numpy as np, sys, os, warnings
from scipy.fftpack import fft2, ifft2
from scipy import ndimage, optimize

def apply_maxmin_normalization(maps):
    min_val = np.nanmin(maps)
    max_val = np.nanmax(maps)
    return (maps - min_val) / (max_val - min_val) 
    
def load_all_moments(filename, bandpass_centers, max_lines=-1):
    moments_data = np.load(filename)[:max_lines]
    moments = {}

    norms = [
        bandpass_centers,        # S2aa
        bandpass_centers,        # S2bb
        bandpass_centers,        # S2ab
        bandpass_centers,        # S3aaa
        bandpass_centers,        # S3bbb
        bandpass_centers,        # S3aab
        bandpass_centers,        # S3abb
        bandpass_centers**2,     # S4aaaa
        bandpass_centers**2,     # S4bbbb
        bandpass_centers**2,     # S4aaab
        bandpass_centers**2,     # S4aabb
        bandpass_centers**2      # S4abbb
    ]

    for i in range(12):
        label = f"moment_{i:02d}"
        moments[label] = [m / norms[i] for m in moments_data[:, :, i]]

    return moments

def get_peak_masks(tmap, mask_threshold_sigma_units = 10, mask_radius_pixel_units = 0, perform_apod = 1, mask_shape = 'circle', taper_radius_fac = 2. ):
    
    """
    Get masking using sigma clipping.

    Parameters
    ----------
    tmap: array
        flatsky map.
    mask_threshold_sigma_units: float
        threshold for sigma clipping.
        mask = np.where( (abs(tmap) > np.mean(tmap) + sigma * np.std(tmap)) )
    mask_radius_pixel_units: float
        punch bigger hole around each peak.
        Default is 0, in which case, only peak pixels satisfying the above condition are masked.
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        circle or a square mask.
        default is circle.
        Must be on of ['circle', 'square']
    taper_radius_fac: float
        radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        default is 2 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    peak_mask: array, shape is tmap.shape.
        binary peak mask.
    mask: array shape is tmap.shape.
        binary or apodised mask with some finite radius around the peaks.
        Only computed when mask_radius_pixel_units>0.
        else, mask_radius_pixel_units == peak_mask
    """

    assert mask_radius_pixel_units >=0

    peak_mask = np.ones( tmap.shape )
    inds = np.where( (abs(tmap) > abs(np.mean(tmap)) + mask_threshold_sigma_units * np.std(tmap)) )
    peak_mask[inds] = 0.


    if mask_radius_pixel_units > 0:
        x_grid, y_grid = np.indices( tmap.shape )

        assert mask_shape in ['circle', 'square']

        mask = np.ones( y_grid.shape )

        #imshow(mask); colorbar(); 
        #pick all the masked inds
        for (i,j) in zip(inds[0], inds[1]):

            y, x = x_grid[j, i], y_grid[j, i]
            
            #plot( y, x, 'ko', color = 'None', mec = 'black', ms = 10, mew = 1.)

            if mask_shape == 'circle':
                radius = np.sqrt( ((x-x_grid)**2. + (y-y_grid)**2.) )
                inds_to_mask = np.where((radius<=mask_radius_pixel_units))
            elif mask_shape == 'square':
                inds_to_mask = np.where( (abs(x-x_grid)<mask_radius_pixel_units) & (abs(y-y_grid)<mask_radius_pixel_units))

            mask[inds_to_mask[0], inds_to_mask[1]] = 0.
        #show(); sys.exit()

        taper_radius = mask_radius_pixel_units * taper_radius_fac #
        if perform_apod:
            ker=np.hanning(taper_radius)
            ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
            mask=ndimage.convolve(mask, ker2d)
            mask/=mask.max()
    else:
        mask = peak_mask

    return peak_mask, mask

def boundary_apod_mask(x_grid, y_grid, mask_radius, perform_apod = True, mask_shape = 'circle', taper_radius_fac = 6.):

    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.

    Parameters
    ----------
    x_grid: array
        x grid of the map (like ra).
    y_grid: array
        y grid of the map (like dec).
    mask_radius: float
        mask radius in same units as x_grid and y_grid
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        circle or a square mask.
        default is circle.
        Must be on of ['circle', 'square']
    taper_radius_fac: float
        radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        default is 6 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    mask: array, shape is x_grid.shape.
        binary or apodised mask.
    """

    assert mask_shape in ['circle', 'square']

    mask = np.ones( y_grid.shape )

    if mask_shape == 'circle':
        radius = np.sqrt( (x_grid**2. + y_grid**2.) )
        inds_to_mask = np.where((radius<=mask_radius))
    elif mask_shape == 'square':
        inds_to_mask = np.where( (abs(x_grid)<mask_radius) & (abs(y_grid)<mask_radius))

    mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    taper_radius = mask_radius * taper_radius_fac #
    if perform_apod:
        ##imshow(mask); colorbar(); show(); sys.exit()
        ker=np.hanning(taper_radius)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
        mask=ndimage.convolve(mask, ker2d)
        mask/=mask.max()

    return mask

def get_mask_using_gaussian_fitting(nonpeak_mask, mul_width_by_factor = 2, ini_height = 0., ini_amp = 1., ini_rot = 0., ini_blob_size_in_pixels = 10., use_elliptical_gaussian = False, perform_apod = True):
        
    #fit Gaussian to the tmap to get the centroid
    ny, nx = nonpeak_mask.shape
    x = y = np.arange(nx)
    xgrid, ygrid = np.meshgrid(x,y)
    #predict initial parameters
    height = ini_height #np.std(nonpeak_mask) #overall floor
    amp = ini_amp #np.max(nonpeak_mask) #peak
    #ellipse rotation
    rot = ini_rot
    #width
    blob_size_in_pixels = ini_blob_size_in_pixels
    wx = wy = blob_size_in_pixels

    non_zero_yinds, non_zero_xinds = np.where( nonpeak_mask == 1)
    howmany = len(non_zero_yinds)

    final_fit_arr = []
    total_mask = np.zeros( nonpeak_mask.shape )
    for cntr, (x,y) in enumerate( zip(non_zero_xinds, non_zero_yinds) ):
        if not (cntr<10 or cntr>howmany-10): continue
        #if not (cntr>howmany-10): continue
        #print(cntr, howmany, x, y)

        #centres
        #y_cen_ini, x_cen_ini = np.unravel_index(np.argmax(tmap), tmap.shape)
        x_cen_ini, y_cen_ini = x, y
        if use_elliptical_gaussian:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx, wy, rot]
        else:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx]

        #bounds - not used currently
        ##lbounds = np.asarray( [height*3., amp*3., x_cen - 10, y_cen - 10, wx/2] )
        ##ubounds = np.asarray( [height*3., amp*3., x_cen + 10, y_cen + 10, wx*2.] )
        p1, success = optimize.leastsq(fitting_func, p0[:], args=(p0, xgrid, ygrid, nonpeak_mask))#, lbounds, ubounds))
        final_fit_arr.append( p1 )
        nonpeak_mask_fit = fitting_func(p1,p1,xgrid,ygrid, nonpeak_mask,return_fit = 1)

        '''
        subplot(121); imshow(nonpeak_mask); colorbar()
        subplot(122); imshow(nonpeak_mask_fit); colorbar(); show(); 
        #sys.exit()
        '''

        #create a mask based on the centres and the widths
        x_fit, y_fit, x_width = p1[2:]
        width_for_mask = x_width * mul_width_by_factor
        rad_grid = np.hypot( xgrid - x_fit, ygrid - y_fit ) 
        inds_to_mask = np.where( rad_grid <= width_for_mask )
        curr_mask = np.zeros( nonpeak_mask.shape )
        curr_mask[inds_to_mask] = 1.
        total_mask = total_mask + curr_mask

    final_mask = np.ones( nonpeak_mask.shape )
    final_mask[total_mask != 0] = 0.
    '''
    subplot(121); imshow(nonpeak_mask); colorbar()
    subplot(122); imshow(total_mask); colorbar(); show(); 
    #sys.exit()
    '''
    if perform_apod:
        pix = 1
        radius = (nx * pix)/10.
        npix_cos = int(radius/pix)
        ker=np.hanning(npix_cos)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )

        final_mask=ndimage.convolve(final_mask, ker2d)
        final_mask/=final_mask.max()
    return final_mask

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter
################################################################################################################

def wiener_filter(mapparams, cl_signal, cl_noise, el = None):

    if el is None:
        el = np.arange(len(cl_signal))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams) 
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams) 

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)

    return wiener_filter

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
    
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)

    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_set = torch.tensor(data[train_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    val_set = torch.tensor(data[val_indices].transpose(0, 3, 1, 2), dtype=torch.float32)
    test_set = torch.tensor(data[test_indices].transpose(0, 3, 1, 2), dtype=torch.float32)

    return train_set, val_set, test_set