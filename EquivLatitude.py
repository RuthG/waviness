import numpy as np


def cell_area(lat_s, lat_e, dlon):
    """
    Calculate area weight for specified lat/lon area.

    Parameters
    ----------
    lat_s : float
        Southernmost latitude
    lat_e : float
        Northernmost latitude
    dlon : float
        Longitude spacing

    Returns
    --------
    wgt : float
        Area weight for single grid box between lat_s -> lat_e and an arbitrary
        lon_s->lon_e (not given)

    """
    rad = np.pi/180.0
    return (dlon*rad)*(np.sin(rad*lat_s)-np.sin(rad*lat_e))


def inv_area(area_total):
    """
    Compute equivalent latitude given an area on a sphere.

    Parameters
    ----------
    area_total : array_like
        Array of areas for each gridpoint

    Returns
    -------
    lateq : array_like
        Latitude equivalent to area swept out by each gridpoint

    """
    radm1 = 180.0/np.pi
    return radm1*np.arcsin(1.0-(area_total/(2.0*np.pi)))


def eqlat(pv, lat, lon, nlev):
    """
    Calculate equivalent latitude from PV field.

    Parameters
    ----------
    pv : array_like
        potential vorticity on some surface (isobaric or isentropic)
    lat : array_like
        latitude locations of pv array
    lon : array_like
        longitude locations of pv array
    N : integer
        number of equivalent latitudes requested

    Returns
    -------
    eqlat : array_like
       Equivalent latitude
    pv_levs : array_like
        N - 2 array of pv levels where EQ lat has been calculated

    """
    rank = len(pv.shape)

    if np.abs(lat[0]) < np.abs(lat[1]-lat[0]):
        lat = np.append(lat[0] - (lat[1]-lat[0]), lat)
    else:
        # Sorry Penny! :)
        raise NotImplementedError('Equiv. lat method NotImplemented for SH [yet]')

    lons, lats = np.meshgrid(lon, lat)
    dlon = lon[1] - lon[0]
    grid_area = cell_area(lats[1:, ...], lats[:-1, ...], dlon)

    # Find min/max pv for each time/level
    pv_min = np.nanmin(np.min(pv, axis=-2), axis=-1)
    pv_max = np.nanmax(np.max(pv, axis=-2), axis=-1)

    if rank == 2:
        pv_levs = np.linspace(pv_min, pv_max, nlev)
        eq_area = np.zeros(nlev)
        for pvidx in range(nlev):
            eq_area[pvidx] = np.ma.masked_where(pv <= pv_levs[pvidx], grid_area).sum()
    elif rank == 3:
        eq_area = np.zeros([pv.shape[0], nlev])
        pv_levs = np.zeros([pv.shape[0], nlev])
        for zidx in range(pv.shape[0]):
            pv_levs[zidx, :] = np.linspace(pv_min[zidx], pv_max[zidx], nlev)
            for pvidx in range(nlev):
                eq_area[zidx, pvidx] = np.ma.masked_where(pv[zidx, ...] <= pv_levs[zidx,
                                                                                   pvidx],
                                                          grid_area).sum()
    elif rank == 4:
        nt, nz, ny, nx = pv.shape
        eq_area = np.zeros([nt, nz, nlev])
        pv_levs = np.zeros([nt, nz, nlev])
        for tidx in range(pv.shape[0]):
            for zidx in range(pv.shape[1]):
                pv_levs[tidx, zidx, :] = np.linspace(pv_min[tidx, zidx],
                                                     pv_max[tidx, zidx], nlev)
                for pvidx in range(nlev-1):
                    eq_area[tidx, zidx, pvidx] = np.ma.masked_where(
                        pv[tidx, zidx, ...] <= pv_levs[tidx, zidx, pvidx],
                        grid_area).sum()
    return np.ma.masked_invalid(inv_area(eq_area)), pv_levs