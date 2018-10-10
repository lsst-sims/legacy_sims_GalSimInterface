import unittest
import numpy as np
import lsst.utils.tests
from lsst.sims.utils import arcsecFromRadians
from lsst.sims.GalSimInterface import GalSimCelestialObject
from lsst.sims.photUtils import Sed, BandpassDict, Bandpass
from lsst.sims.photUtils import PhotometricParameters


def setup_module(module):
    lsst.utils.tests.init()


class GSCOTestCase(unittest.TestCase):

    def test_getters(self):
        """
        Verify that what goes into __init__ comes out of the getters
        """
        xpupil_rad = 2.176
        ypupil_rad = 3.2112
        hlr = 1.632
        minor_axis_rad = 9.124
        major_axis_rad = 21.2684
        position_angle_rad = 0.1334
        sindex = 4.387
        npoints = 19
        pixel_scale=25.134
        rotation_angle_rad = 0.9335
        gamma1 = 0.512
        gamma2 = 3.4456
        kappa = 1.747
        phot_params = PhotometricParameters()

        rng = np.random.RandomState(88)
        wav = np.arange(0.1, 200.0, 0.17)
        spec = Sed(wavelen=wav, flambda=rng.random_sample(len(wav)))

        # copy spec for comparison below
        spec2 = Sed(wavelen=spec.wavelen, flambda=spec.flambda)

        # keep BandpassDict on the same wavelength grid as Sed,
        # otherwise, the random initialization results in flux==NaN
        bp_list = []
        bp_name_list = []
        for bp_name in 'abcd':
            sb = rng.random_sample(len(wav))
            bp = Bandpass(wavelen=wav, sb=sb)
            bp_list.append(bp)
            bp_name_list.append(bp_name)

        bp_dict = BandpassDict(bp_list, bp_name_list)

        gso = GalSimCelestialObject('pointSource',
                                    xpupil_rad, ypupil_rad,
                                    hlr, minor_axis_rad, major_axis_rad,
                                    position_angle_rad, sindex,
                                    spec, bp_dict, phot_params,
                                    npoints, 'bob', pixel_scale,
                                    rotation_angle_rad,
                                    gamma1=gamma1, gamma2=gamma2,
                                    kappa=kappa, uniqueId=111)

        self.assertAlmostEqual(gso.xPupilRadians/xpupil_rad, 1.0, 10)
        self.assertAlmostEqual(gso.xPupilArcsec/arcsecFromRadians(xpupil_rad), 1.0, 10)
        self.assertAlmostEqual(gso.yPupilRadians/ypupil_rad, 1.0, 10)
        self.assertAlmostEqual(gso.yPupilArcsec/arcsecFromRadians(ypupil_rad), 1.0, 10)
        self.assertAlmostEqual(gso.halfLightRadiusRadians/hlr, 1.0, 10)
        self.assertAlmostEqual(gso.halfLightRadiusArcsec/arcsecFromRadians(hlr), 1.0, 10)
        self.assertEqual(gso.uniqueId, 111)
        self.assertEqual(gso.galSimType, 'pointSource')
        self.assertEqual(gso.npoints, npoints)
        self.assertAlmostEqual(gso.minorAxisRadians/minor_axis_rad, 1.0, 10)
        self.assertAlmostEqual(gso.majorAxisRadians/major_axis_rad, 1.0, 10)
        self.assertAlmostEqual(gso.positionAngleRadians/position_angle_rad, 1.0, 10)
        self.assertAlmostEqual(gso.sindex/sindex, 1.0, 10)
        self.assertAlmostEqual(gso.pixel_scale/pixel_scale, 1.0, 10)
        self.assertAlmostEqual(gso.rotation_angle/rotation_angle_rad, 1.0, 10)
        g1 = gamma1/(1.0-kappa)
        self.assertAlmostEqual(gso.g1/g1, 1.0, 10)
        g2 = gamma2/(1.0-kappa)
        self.assertAlmostEqual(gso.g2/g2, 1.0, 10)
        mu = 1.0/((1.0-kappa)**2-(gamma1**2+gamma2**2))
        self.assertAlmostEqual(gso.mu/mu, 1.0, 10)
        self.assertEqual(gso.fits_image_file, 'bob')
        self.assertEqual(gso.sed, spec2)
        for bp_name in bp_dict:
            ff = spec2.calcADU(bp_dict[bp_name], phot_params)
            ff *= phot_params.gain
            self.assertTrue(np.isfinite(ff))
            self.assertAlmostEqual(gso.flux(bp_name)/ff, 1.0, 10)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
