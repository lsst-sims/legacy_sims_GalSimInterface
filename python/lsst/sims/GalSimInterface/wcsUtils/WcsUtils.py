import numpy

__all__ = ["_nativeLonLatFromRaDec", "_raDecFromNativeLonLat",
           "nativeLonLatFromRaDec", "raDecFromNativeLonLat"]


def _nativeLonLatFromRaDec(ra, dec, raPointing, decPointing):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The coordinate basis axes for this system is achieved by taking the true
    coordinate basis axes and rotating them once about the z axis and once about
    the x axis (or, by rotating the vector pointing to the RA and Dec being
    transformed once about the x axis and once about the z axis).  These
    are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    @param [in] ra is the RA of the star being transformed in radians

    @param [in] dec is the Dec of the star being transformed in radians

    @param [in] raPointing is the RA at which the telescope is pointing
    in radians

    @param [in] decPointing is the Dec at which the telescope is pointing
    in radians

    @param [out] lonOut is the native longitude in radians

    @param [out] latOut is the native latitude in radians
    """

    x = -1.0*numpy.cos(dec)*numpy.sin(ra)
    y = numpy.cos(dec)*numpy.cos(ra)
    z = numpy.sin(dec)

    alpha = decPointing - 0.5*numpy.pi
    beta = -1.0*raPointing

    ca=numpy.cos(alpha)
    sa=numpy.sin(alpha)
    cb=numpy.cos(beta)
    sb=numpy.sin(beta)

    v2 = numpy.dot(numpy.array([
                                [1.0, 0.0, 0.0],
                                [0.0, ca, sa],
                                [0.0, -1.0*sa, ca]
                                ]),
                   numpy.dot(numpy.array([[cb, -1.0*sb, 0.0],
                                          [sb, cb, 0.0],
                                          [0.0, 0.0, 1.0]
                                          ]), numpy.array([x,y,z])))

    cc = numpy.sqrt(v2[0]*v2[0]+v2[1]*v2[1])
    latOut = numpy.arctan2(v2[2], cc)

    if numpy.abs(latOut)>numpy.pi/2.0:
        raise RuntimeError('returned latOut %e' % latOut)

    _y = v2[1]/numpy.cos(latOut)
    _ra = numpy.arccos(_y)
    _x = -numpy.sin(_ra)

    if numpy.isnan(_ra):
        lonOut = 0.0
    elif (numpy.abs(_x)>1.0e-9 and numpy.sign(_x)!=numpy.sign(v2[0])) \
         or (numpy.abs(_y)>1.0e-9 and numpy.sign(_y)!=numpy.sign(v2[1])):
        lonOut = 2.0*numpy.pi-_ra
    else:
        lonOut = _ra

    return lonOut, latOut


def nativeLonLatFromRaDec(ra, dec, raPointing, decPointing):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The coordinate basis axes for this system is achieved by taking the true
    coordinate basis axes and rotating them once about the z axis and once about
    the x axis (or, by rotating the vector pointing to the RA and Dec being
    transformed once about the x axis and once about the z axis).  These
    are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    @param [in] ra is the RA of the star being transformed in degrees

    @param [in] dec is the Dec of the star being transformed in degrees

    @param [in] raPointing is the RA at which the telescope is pointing
    in degrees

    @param [in] decPointing is the Dec at which the telescope is pointing
    in degrees

    @param [out] lonOut is the native longitude in degrees

    @param [out] latOut is the native latitude in degrees
    """

    lon, lat = _nativeLonLatFromRaDec(numpy.radians(ra), numpy.radians(dec),
                                      numpy.radians(raPointing),
                                      numpy.radians(decPointing))

    return numpy.degrees(lon), numpy.degrees(lat)


def _raDecFromNativeLonLat(lon, lat, raPointing, decPointing):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for _nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in radians

    @param [in] lat is the native latitude in radians

    @param [in] raPointing is the RA at which the telescope is pointing in
    radians

    @param [in] decPointing is the Dec at which the telescope is pointing
    in radians

    @param [out] raOut is the RA of the star in radians

    @param [in] decOut is the Dec of the star in radians
    """

    x = -1.0*numpy.cos(lat)*numpy.sin(lon)
    y = numpy.cos(lat)*numpy.cos(lon)
    z = numpy.sin(lat)

    alpha = 0.5*numpy.pi - decPointing
    beta = raPointing

    ca=numpy.cos(alpha)
    sa=numpy.sin(alpha)
    cb=numpy.cos(beta)
    sb=numpy.sin(beta)

    v2 = numpy.dot(numpy.array([[cb, -1.0*sb, 0.0],
                                [sb, cb, 0.0],
                                [0.0, 0.0, 1.0]
                                ]),
                                numpy.dot(numpy.array([[1.0, 0.0, 0.0],
                                                       [0.0, ca, sa],
                                                       [0.0, -1.0*sa, ca]
                                ]),
                                numpy.array([x,y,z])))


    cc = numpy.sqrt(v2[0]*v2[0]+v2[1]*v2[1])
    decOut = numpy.arctan2(v2[2], cc)

    if numpy.abs(decOut)>numpy.pi/2.0:
        raise RuntimeError('returned decOut %e' % decOut)

    _y = v2[1]/numpy.cos(decOut)
    _ra = numpy.arccos(_y)
    _x = -numpy.sin(_ra)

    if numpy.isnan(_ra):
        raOut = 0.0
    elif (numpy.abs(_x)>1.0e-9 and numpy.sign(_x)!=numpy.sign(v2[0])) \
         or (numpy.abs(_y)>1.0e-9 and numpy.sign(_y)!=numpy.sign(v2[1])):
        raOut = 2.0*numpy.pi-_ra
    else:
        raOut = _ra

    return raOut, decOut


def raDecFromNativeLonLat(lon, lat, raPointing, decPointing):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in degrees

    @param [in] lat is the native latitude in degrees

    @param [in] raPointing is the RA at which the telescope is pointing in
    degrees

    @param [in] decPointing is the Dec at which the telescope is pointing
    in degrees

    @param [out] raOut is the RA of the star in degrees

    @param [in] decOut is the Dec of the star in degrees
    """

    ra, dec = _raDecFromNativeLonLat(numpy.radians(lon),
                                     numpy.radians(lat),
                                     numpy.radians(raPointing),
                                     numpy.radians(decPointing))

    return numpy.degrees(ra), numpy.degrees(dec)
