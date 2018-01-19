from builtins import object
import numpy
from lsst.sims.utils import arcsecFromRadians

__all__ = ["GalSimCelestialObject"]

class GalSimCelestialObject(object):
    """
    This is a class meant to carry around all of the data required by
    the GalSimInterpreter to draw an image of a single object.  The idea
    is that all of the drawing functions in the GalSimInterpreter will
    just take a GalSimCelestialObject as an argument, rather than taking
    a bunch of different arguments, one for each datum.
    """

    def __init__(self, uniqueId, galSimType, sed, ra, dec, xPupil, yPupil,
                 halfLightRadius, minorAxis, majorAxis, positionAngle,
                 sindex, fluxDict, gamma1=0, gamma2=0, kappa=0):
        """
        @param [in] uniqueId is an int storing a unique identifier for this object

        @param [in] galSimType is a string, either 'pointSource', 'sersic' or 'RandomWalk' denoting the shape of the object

        @param [in] sed is the SED of the object (an instantiation of the Sed class defined in
        sims_photUtils/../../Sed.py

        @param [in] ra is the RA coordinate of the object in radians
        (ICRS)

        @param [in] dec is the Dec coordinate of the object in radians
        (ICRS)

        @param [in] xPupil is the x pupil coordinate of the object in radians

        @param [in] yPupil is the y pupil coordinate of the object in radians

        @param [in] halfLightRadius is the halfLightRadius of the object in radians

        @param [in] minorAxis is the semi-minor axis of the object in radians

        @param [in] majorAxis is the semi-major axis of the object in radians

        @param [in] positionAngle is the position angle of the object in radians

        @param [in] sindex is the sersic index of the object

        @param [in] fluxDict is a dict of electron count (not ADU) values keyed to bandpass names,
        i.e. {'u':44000, 'g':41000} would mean the source produces 44000 electrons in the
        u band and 41000 electrons in the g band.

        @param [in] gamma1 is the real part of the WL shear parameter

        @param [in] gamma2 is the imaginary part of the WL shear parameter

        @param [in] kappa is the WL convergence parameter
        """
        self._uniqueId = uniqueId
        self._galSimType = galSimType
        self._sed = sed
        self._raRadians = ra
        self._decRadians = dec
        self._xPupilRadians = xPupil
        self._xPupilArcsec = arcsecFromRadians(xPupil)
        self._yPupilRadians = yPupil
        self._yPupilArcsec = arcsecFromRadians(yPupil)
        self._halfLightRadiusRadians = halfLightRadius
        self._halfLightRadiusArcsec = arcsecFromRadians(halfLightRadius)
        self._minorAxisRadians = minorAxis
        self._majorAxisRadians = majorAxis
        self._positionAngleRadians = positionAngle
        self._sindex = sindex
        # The galsim.lens(...) function wants to be passed reduced
        # shears and magnification, so convert the WL parameters as
        # defined in phosim instance catalogs to these values.  See
        # https://github.com/GalSim-developers/GalSim/blob/releases/1.4/doc/GalSim_Quick_Reference.pdf
        # and Hoekstra, 2013, http://lanl.arxiv.org/abs/1312.5981
        self._g1 = gamma1/(1. - kappa)   # real part of reduced shear
        self._g2 = gamma2/(1. - kappa)   # imaginary part of reduced shear
        self._mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        self._fluxDict = fluxDict

    @property
    def uniqueId(self):
        return self._uniqueId

    @uniqueId.setter
    def uniqueId(self, value):
        raise RuntimeError("You should not be setting the unique id on the fly; " \
                           + "just instantiate a new GalSimCelestialObject")

    @property
    def galSimType(self):
        return self._galSimType

    @galSimType.setter
    def galSimType(self, value):
        raise RuntimeError("You should not be setting galSimType on the fly; " \
                           + "just instantiate a new GalSimCelestialObject")

    @property
    def sed(self):
        return self._sed

    @sed.setter
    def sed(self, value):
        raise RuntimeError("You should not be setting sed on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def raRadians(self):
        return self._raRadians

    @raRadians.setter
    def raRadians(self, value):
        raise RuntimeError("You should not be setting raRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def decRadians(self):
        return self._decRadians

    @decRadians.setter
    def decRadians(self, value):
        raise RuntimeError("You should not be setting decRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def xPupilRadians(self):
        return self._xPupilRadians

    @xPupilRadians.setter
    def xPupilRadians(self, value):
        raise RuntimeError("You should not be setting xPupilRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def xPupilArcsec(self):
        return self._xPupilArcsec

    @xPupilArcsec.setter
    def xPupilArcsec(self, value):
        raise RuntimeError("You should not be setting xPupilArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def yPupilRadians(self):
        return self._yPupilRadians

    @yPupilRadians.setter
    def yPupilRadians(self, value):
        raise RuntimeError("You should not be setting yPupilRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def yPupilArcsec(self):
        return self._yPupilArcsec

    @yPupilArcsec.setter
    def yPupilArcsec(self, value):
        raise RuntimeError("You should not be setting yPupilArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def halfLightRadiusRadians(self):
        return self._halfLightRadiusRadians

    @halfLightRadiusRadians.setter
    def halfLightRadiusRadians(self, value):
        raise RuntimeError("You should not be setting halfLightRadiusRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def halfLightRadiusArcsec(self):
        return self._halfLightRadiusArcsec

    @halfLightRadiusArcsec.setter
    def halfLightRadiusArcsec(self, value):
        raise RuntimeError("You should not be setting halfLightRadiusArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def minorAxisRadians(self):
        return self._minorAxisRadians

    @minorAxisRadians.setter
    def minorAxisRadians(self, value):
        raise RuntimeError("You should not be setting minorAxisRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def majorAxisRadians(self):
        return self._majorAxisRadians

    @majorAxisRadians.setter
    def majorAxisRadians(self, value):
        raise RuntimeError("You should not be setting majorAxisRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def positionAngleRadians(self):
        return self._positionAngleRadians

    @positionAngleRadians.setter
    def positionAngleRadians(self, value):
        raise RuntimeError("You should not be setting positionAngleRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def sindex(self):
        return self._sindex

    @sindex.setter
    def sindex(self, value):
        raise RuntimeError("You should not be setting sindex on the fly; " \
        + "just instantiate a new GalSimCelestialObject")

    @property
    def g1(self):
        return self._g1

    @g1.setter
    def g1(self, value):
        raise RuntimeError("You should not be setting g1 on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def g2(self):
        return self._g2

    @g2.setter
    def g2(self, value):
        raise RuntimeError("You should not be setting g2 on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        raise RuntimeError("You should not be setting mu on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    def flux(self, band):
        """
        @param [in] band is the name of a bandpass

        @param [out] the ADU in that bandpass, as stored in self._fluxDict
        """
        if band not in self._fluxDict:
            raise RuntimeError("Asked GalSimCelestialObject for flux in %s; that band does not exist" % band)

        return self._fluxDict[band]
