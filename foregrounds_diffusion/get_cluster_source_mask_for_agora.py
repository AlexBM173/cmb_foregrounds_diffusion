def get_mdpl2_halo_cat(get_velocities = True):
    halo_cat_fname = 'haloc/v122921/halos_extracted_with_velocities_refined_minM500c1e+13_maxM500c1e+20_minz0.0_maxz4.0.npz.npy'


    halo_cat = np.load(halo_cat_fname, allow_pickle=True)
    mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c, mdpl2_vlos, mdpl2_vtht, mdpl2_vphi = halo_cat.T


    if get_velocities:
        return mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c, mdpl2_vlos, mdpl2_vtht, mdpl2_vphi
    else:
        return mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c




def get_cluster_mask_radius(m500c):


    """
    please change this as you like -- based on experiment beam, etc. --. Generally should be fine for a 1 arcmin experiment.
    """
    
    if m500c<1e14:
        cluster_mask_rad = 3.
    elif m500c>=1e14 and m500c<3e14:
        cluster_mask_rad = 5.
    elif m500c>=3e14 and m500c<5e14:
        cluster_mask_rad = 8.
    else:
        cluster_mask_rad = 10.


    return cluster_mask_rad


def get_point_source_mask_in_healpix(freq, hmap_Mjy_per_sr, threshold_mjy_freq0, threshold2_mjy_freq0 = None, freq0 = 150., spec_index = 3.4, full_sky = True, ang_res_am = None, return_flux_map_in_mjy = False):


    if full_sky:
        nside = H.get_nside(hmap_Mjy_per_sr)
        pix_area = H.nside2resol(nside)**2.
    else:
        assert ang_res_am is not None
        ang_res_rad = np.radians(ang_res_am/60.)
        pix_area = ang_res_rad**2.


    hmap_Mjy = np.copy( hmap_Mjy_per_sr ) * pix_area
    MJy_mJy = 1e9
    hmap_mjy = hmap_Mjy * MJy_mJy
    ###imshow(hmap_mjy, vmax = threshold_mjy_freq0); colorbar(); show(); sys.exit()


    scaling = (freq / freq0) ** spec_index
    threshold_mjy = threshold_mjy_freq0 * scaling
    ###print(threshold_mjy, threshold_mjy_freq0); sys.exit()


    if threshold2_mjy_freq0 is None:
        mask_pixels = np.where(hmap_mjy >= threshold_mjy)
    else:
        threshold2 = threshold2_mjy_freq0 * scaling
        mask_pixels = np.where( (hmap_mjy >= threshold_mjy) & (hmap_mjy<threshold2) )


    if full_sky:
        mask_pixels = mask_pixels[0]


    if return_flux_map_in_mjy:
        return mask_pixels, hmap_mjy
    else:
        return mask_pixels




def get_apodised_mdpl2_cluster_mask(nside, m500c_threshold = 2e14, cluster_lmz_dic = None, howmanythetaforclusters = -1, apodise = True, store_mask = True, expname = None): #tsz mask


    import copy
    print('\n\tget apodised cluster mask.')


    #change this Omar. Use your Agora ra/dec/mass/redshifts. You do not need the vecloity entries.
    mdpl2_ra, mdpl2_dec, mdpl2_z, mdpl2_m200c, mdpl2_m500c, mdpl2_vlos, mdpl2_vtht, mdpl2_vphi = get_mdpl2_halo_cat()


    if m500c_threshold != -1:
        clus_inds = np.where(mdpl2_m500c>=m500c_threshold)[0]
    else: #get cluster mask based on redshift
        assert expname is not None
        print('\tget cluster mask based on redshift')
        redshifts = cluster_lmz_dic['redshift']
        dz = np.diff(redshifts)[0]
        lim_M500c = cluster_lmz_dic['M500c'] * 1e14
        clus_inds = []
        if (1): #z = 0 to z = 0.1
            inds = np.where( (mdpl2_z<redshifts[0]) )[0]
            passed_inds = np.where(mdpl2_m500c[inds]>lim_M500c[0])[0]
            detected_mass_inds = inds[passed_inds]
            clus_inds.extend( detected_mass_inds )
        for zcntr, zzz in enumerate( redshifts ):
            inds = np.where( (mdpl2_z>=zzz) & (mdpl2_z<zzz+dz))[0]
            curr_lim_M500c = lim_M500c[zcntr]
            passed_inds = np.where(mdpl2_m500c[inds]>curr_lim_M500c)[0]
            detected_mass_inds = inds[passed_inds]


            if len(detected_mass_inds)>0:
                actual_det_M500_in_cat = min( mdpl2_m500c[detected_mass_inds] ) / 1e14
            else:                
                actual_det_M500_in_cat = -1


            print('\t\t\tz = %s; total masked = %s; Min masses: Limit = %s; Actual = %s' %(zzz, len(detected_mass_inds), curr_lim_M500c/1e14, actual_det_M500_in_cat))
            clus_inds.extend( detected_mass_inds )


    ###print( len(clus_inds) ); sys.exit()
    print('\t\ttotal clusters to be masked = %s' %(len(clus_inds))); ##sys.exit()


    if just_return_masked_inds:
        clus_inds = np.asarray(clus_inds)
        return clus_inds, mdpl2_ra[clus_inds], mdpl2_dec[clus_inds], mdpl2_m500c[clus_inds]


    h, omega_m = 0.6774, 0.3089
    cosmo = FlatLambdaCDM(H0 = h*100., Om0 = omega_m)


    if howmanythetaforclusters != -1: #get theta500c now
        cluster_mask_radius_am_arr = []        
        for cntr, iii in enumerate( clus_inds ):
            if cntr%1000==0: print(cntr)
            tmpc500c = concentration.concentration(mdpl2_m500c[iii], '500c', mdpl2_z[iii])
            m500cval, r500cval, c500cval = mass_defs.changeMassDefinition(mdpl2_m500c[iii], tmpc500c, mdpl2_z[iii], '500c', '500c', profile='nfw')
            r500cval_mpc = r500cval/1e3
            
            ang_dia_dist = cosmo.comoving_distance(mdpl2_z[iii])/(1.+mdpl2_z[iii])


            #from IPython import embed; embed(); sys.exit()
            theta500cval_am = np.degrees( r500cval_mpc/ang_dia_dist.value ) * 60.


            cluster_mask_radius_am = int( theta500cval_am * howmanythetaforclusters )+1
            cluster_mask_radius_am_arr.append( cluster_mask_radius_am )
            ##print(len(cluster_mask_radius_am_arr))


        if (1): #refined cluster_mask_radius_am_arr to few set of radii
            cluster_mask_radius_am_arr = np.asarray(cluster_mask_radius_am_arr)
            cluster_mask_radius_am_arr_mod = np.zeros_like(cluster_mask_radius_am_arr)
            cluster_mask_radius_am_arr_mod[cluster_mask_radius_am_arr<=5.] = 5.
            cluster_mask_radius_am_arr_mod[(cluster_mask_radius_am_arr>5.) & (cluster_mask_radius_am_arr<=10.)] = 8.
            cluster_mask_radius_am_arr_mod[(cluster_mask_radius_am_arr>10.) & (cluster_mask_radius_am_arr<=20.)] = 15.
            cluster_mask_radius_am_arr_mod[(cluster_mask_radius_am_arr>20.) & (cluster_mask_radius_am_arr<=50.)] = 35.
            cluster_mask_radius_am_arr_mod[(cluster_mask_radius_am_arr>50.) & (cluster_mask_radius_am_arr<=100.)] = 75.
            cluster_mask_radius_am_arr_mod[cluster_mask_radius_am_arr>100] = 100.
            cluster_mask_radius_am_arr = np.copy(cluster_mask_radius_am_arr_mod)


    #create different sets based on cluster masking radius
    print('\n\tcreate different sets based on cluster masking radius')
    npix = H.nside2npix(nside)
    hmask_dic = {}    
    for cntr, iii in enumerate( clus_inds ):
        if cntr%5000 ==0:print(cntr)
        ppp = H.ang2pix(nside, np.radians(90.-mdpl2_dec[iii]), np.radians(mdpl2_ra[iii]))
        if howmanythetaforclusters != -1:
            cluster_mask_radius_am = cluster_mask_radius_am_arr[cntr]
        else:
            cluster_mask_radius_am = get_cluster_mask_radius(mdpl2_m500c[iii])


        ivec = H.pix2vec(nside, ppp)
        disc = H.query_disc(nside, ivec, np.deg2rad(cluster_mask_radius_am/60.))
        if cluster_mask_radius_am not in hmask_dic:
            hmask_dic[cluster_mask_radius_am] = np.ones(npix)
        hmask_dic[cluster_mask_radius_am][disc] = 0.


    print(hmask_dic.keys())
    print('\t\tbinary masks obtained')
    #np.save('tsz_binary_mask.npy', hmask_dic)
    print('\n\t\tdone'); ###sys.exit()


    if apodise: #now apodise
        print('\n\tnow apodise')
        hmask_smoothed_dic = {}
        for cluster_mask_radius_am in sorted( hmask_dic ):
            print('\n\t\tmask radius = %s' %(cluster_mask_radius_am))


            if cluster_mask_radius_am<=10.:
                apod_angle_am = 10. ##cluster_mask_radius_am * 20. #10.
            else:
                apod_angle_am = 20. ##cluster_mask_radius_am * 20. #10.
            apod_angle = np.radians(apod_angle_am/60.)


            dist_smooth_angle_am = cluster_mask_radius_am ##* 3. #5.
            apod_start_dist_am = 0. ##cluster_mask_radius_am ##0.0


            dist_smooth_angle = np.radians(dist_smooth_angle_am/60.)
            apod_start_dist = np.radians(apod_start_dist_am/60.)
            apod_end_dist = apod_start_dist + apod_angle


            curr_mask = hmask_dic[cluster_mask_radius_am]
            curr_mask_smoothed = apodize_binary_mask_prof(curr_mask, dist_smooth_angle, apod_start_dist, apod_end_dist)
            curr_mask_smoothed = curr_mask_smoothed / np.max(curr_mask_smoothed)
            hmask_smoothed_dic[cluster_mask_radius_am] = curr_mask_smoothed
            #np.save('tsz_apodised_mask.npy', hmask_smoothed_dic)
        print('\n\t\tdone')
    else:
        hmask_smoothed_dic = copy.deepcopy(hmask_dic)


    print('\n\tcreate final mask')
    final_hmask_arr = list( hmask_smoothed_dic.values() )
    final_hmask = np.product( final_hmask_arr , axis = 0)
    final_hmask = final_hmask / np.max(final_hmask)


    return final_hmask




#apod mask
def apodize_binary_mask_prof(
    binary_mask, dist_smooth_angle, apod_start_dist, apod_end_dist
):
    """
    Apodize a binary mask by applying a smooth profile near the boundaries.


    The profile is [x-sin(x)]/2pi in the range [0, 2pi], which is an integral
    of a cross section of the cosine kernel used in the method above, i.e.,
    apodize_binary_mask_conv. (The cross section itself is [1 + cos(x)]/2 in
    the range [-pi, pi].)


    In this method, first, a distance map is created based on the binary mask.
    In this distance map, the value of a pixel represents the distance between
    the corresponding pixel of the binary mask and the nearest masked pixel
    (value 0.0) of the mask. Then, this distance map is smoothed by a Gaussian
    kernel. Finally, the aforementioned profile is applied to the region of the
    binary mask whose counterpart in the smoothed distance map has distance
    values within a certain range.


    If the binary mask represents a field, then the method
    apodize_binary_mask_conv seems to be better than this one because the
    former seems to create a smoother mask in the sense that there is less
    power in the spectrum of the mask beyond ell of a few thousands.


    On the other hand, if the binary mask represents point sources,
    then this latter method seems to create a better-looking apodization mask
    in the sense that the smooth region around each disk masking a source looks
    more circularly symmetric for a high Nside.


    Parameters:
    -----------
    binary_mask : array
        An array that corresponds to a full-sky HEALPix map representing a
        binary mask.
    dist_smooth_angle : float
        The FWHM of the Gaussian kernel used to smooth the distance map.
        G3Units need to be used when this parameter is specified.
    apod_start_dist : float
        The region of the binary mask whose counterpart in the smoothed
        distance map has distance values smaller than the value specified by
        this parameter will not get the profile applied. G3Units need to be
        used when this parameter is specified.
    apod_end_dist : float
        The region of the binary mask whose counterpart in the smoothed
        distance map has distance values larger than the value specified by
        this parameter will not get the profile applied. G3Units need to be
        used when this parameter is specified.


    Returns:
    --------
    smooth_mask : array:
        An array that corresponds to a full-sky HEALPix map representing the
        apodized mask.
    """
    import healpy as hp


    '''
    dist_smooth_angle /= core.G3Units.rad
    apod_start_dist /= core.G3Units.rad
    apod_end_dist /= core.G3Units.rad
    '''
    net_dist = apod_end_dist - apod_start_dist


    # Construct the distance map and smooth it
    dist_map = hp.dist2holes(binary_mask)
    binary_mask[dist_map <= apod_start_dist] = 0.0


    smooth_region = (dist_map > apod_start_dist) & (dist_map < apod_end_dist)
    #20220510 - too slow for 3*nside-1
    #dist_map = hp.smoothing(dist_map, fwhm=dist_smooth_angle)
    nside = H.get_nside(binary_mask)
    dist_map = hp.smoothing(dist_map, fwhm=dist_smooth_angle, lmax = nside)


    # Apply the profile and take care of a few small things
    smooth_mask = np.array(binary_mask)
    x = (dist_map[smooth_region] - apod_start_dist) / net_dist * 2.0 * np.pi
    del dist_map
    smooth_mask[smooth_region] = (x - np.sin(x)) / (2.0 * np.pi)


    smooth_mask *= binary_mask
    smooth_mask[smooth_mask < 0.0] = 0.0
    smooth_mask[smooth_mask > 1.0] = 1.0


    del binary_mask
    return smooth_mask


import numpy as np, sys, os, healpy as H, glob


########################################
########################################
########################################
#### Cluster mask


if (1): #mdpl2 - M500c vs z mask for SPT 100d megadeep


    m500c_threshold = -1
    expname = 'SPT-3G'
    cluster_lmz_dic_fname = 'cluster_limiting_masses_M500c_vs_z.npy'
    cluster_lmz_dic = pickle.load(open(cluster_lmz_dic_fname, 'rb'))[expname]


    nside = 8192
    hmask = get_apodised_mdpl2_cluster_mask(nside, m500c_threshold = m500c_threshold, cluster_lmz_dic = cluster_lmz_dic, expname = expname)#, cluster_mask_radius_am = cluster_mask_radius_am)
    sys.exit()




if (1): #mdpl2 - fixed mass
    nside = 8192
    hmask = get_apodised_mdpl2_cluster_mask(nside, m500c_threshold = 5e13, cluster_lmz_dic = None) #tsz mask




########################################
########################################
########################################
#### Source mask


def get_mdpl2_conversion_factors_K_to_MjyperSr(expname, band):
    mdpl2_MjyperSr_to_K_conv_planck_dic = {100: 243.623, 143: 371.036, 217: 481.882, 353: 287.281, 545: 57.6963, 857: 2.26476}
    mdpl2_MjyperSr_to_K_conv_spt_dic = {95: 208.973, 150: 375.876, 220: 472.522, 221: 473.332, 285: 414.977, 286: 414.977, 345: 310.827}    
    mdpl2_MjyperSr_to_K_conv_s4_dic = {145: 379.391*0.976, 155: 403.379*0.975}
    if expname is None: #20230218
        #assuming same factors for 285 and 286 GHz bands.
        mdpl2_MjyperSr_to_K_conv_dic = {95: 208.973, 150: 375.876, 220: 472.522, 285: 414.977, 345: 310.827}    
    if expname == 'planck':
        curr_dic = mdpl2_MjyperSr_to_K_conv_planck_dic
    elif expname == 'spt3g' or expname == 'spt' or expname == 'spt4':
        curr_dic = mdpl2_MjyperSr_to_K_conv_spt_dic
    elif expname == 'cmbs4' or expname == 's4wide' or expname == 's4deep':
        curr_dic = mdpl2_MjyperSr_to_K_conv_s4_dic
    elif expname is None:
        curr_dic = mdpl2_MjyperSr_to_K_conv_dic


    return curr_dic[band]


#this is the logic
hmap_cib_in_uK = H.read_map( hmap_cib_fname ) #get CIB map at 150 GHz
hmap_rad_in_uK = H.read_map( hmap_rad_fname ) #get Radio map at 150 GHz
hmap_sources_in_uK = hmap_cib_in_uK + hmap_rad_in_uK #combine them


#convert into flux units
band0 = 150
threshold_mjy_band0 = 6. #mJy
conv_factor_K_to_mjyperSr = get_mdpl2_conversion_factors_K_to_MjyperSr(expname, band0) #conversion from K to mJy/Sr
#print(conv_factor_K_to_mjyperSr)
uK_to_K = 1./1e6
hmap_sources_in_mjy_per_sr = np.copy(curr_hmap) * uK_to_K * conv_factor_K_to_mjyperSr


masked_pixels = get_point_source_mask_in_healpix(band0, hmap_sources_in_mjy_per_sr, threshold_mjy_band0)
hmask = np.ones( len(hmap_sources_in_uK) )
hmask[masked_pixels] = 0.