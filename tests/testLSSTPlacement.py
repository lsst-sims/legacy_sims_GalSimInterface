"""
This test mimics testPlacement.py, but is designed to specifically
test the placement of objects on the LSST camera.  This is to make
sure that we have correctly handled the band-dependent optical distortions
in the LSST camera.
"""
from builtins import range
import unittest
import tempfile
import shutil
import lsst.utils.tests

import numpy as np
import os
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from testUtils import create_text_catalog

from lsst.sims.coordUtils import pixelCoordsFromRaDec
from lsst.sims.coordUtils import DMtoCameraPixelTransformer
from lsst.sims.coordUtils import raDecFromPixelCoordsLSST
from lsst.sims.coordUtils import pixelCoordsFromRaDecLSST
from lsst.sims.coordUtils import chipNameFromPupilCoordsLSST
from lsst.sims.coordUtils import pupilCoordsFromFocalPlaneCoordsLSST
from lsst.sims.coordUtils import focalPlaneCoordsFromPupilCoordsLSST
from lsst.sims.coordUtils import lsst_camera

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class LSSTPlacementFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float),
               ('magNorm', 'mag_norm', np.float)]


class LSSTPlacementCatalog(GalSimStars):

    def get_galacticAv(self):
        ra = self.column_by_name('raJ2000')
        return np.array([0.1]*len(ra))

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str, 12)),
                        ('properMotionRa', 0.0, np.float),
                        ('properMotionDec', 0.0, np.float),
                        ('radialVelocity', 0.0, np.float),
                        ('parallax', 0.0, np.float)
                        ]


class GalSimPlacementTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()
        if hasattr(chipNameFromPupilCoordsLSST, '_detector_arr'):
            del chipNameFromPupilCoordsLSST._detector_arr
        if hasattr(focalPlaneCoordsFromPupilCoordsLSST, '_z_fitter'):
            del focalPlaneCoordsFromPupilCoordsLSST._z_fitter
        if hasattr(pupilCoordsFromFocalPlaneCoordsLSST, '_z_fitter'):
            del pupilCoordsFromFocalPlaneCoordsLSST._z_fitter
        if hasattr(lsst_camera, '_lsst_camera'):
            del lsst_camera._lsst_camera

    @classmethod
    def setUpClass(cls):
        opsimdb = os.path.join(getPackageDir('sims_data'), 'OpSimData',
                               'opsimblitz1_1133_sqlite.db')
        obs_gen = ObservationMetaDataGenerator(opsimdb)
        cls.obs_dict = {}
        for band in 'ugrizy':
            obs_list = obs_gen.getObservationMetaData(telescopeFilter=band, limit=10)
            assert len(obs_list) > 0
            cls.obs_dict[band] = obs_list[0]


    def testObjectPlacement(self):
        """
        Test that GalSim places objects on the correct pixel by drawing
        images containing single objects and no background, reading those
        images back in, and comparing the flux-averaged centroids of the
        images with the expected pixel positions of the input objects.
        """
        scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='testLSSTObjectPlacement-')
        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)
        os.mkdir(scratchDir)

        detector = lsst_camera()['R:0,3 S:2,2']
        det_name = 'R03_S22'

        magNorm = 19.0

        pixel_transformer = DMtoCameraPixelTransformer()

        for band in 'ugrizy':
            obs = self.obs_dict[band]

            catName = os.path.join(scratchDir, 'placementCatalog.dat')
            imageRoot = os.path.join(scratchDir, 'placementImage')
            dbFileName = os.path.join(scratchDir, 'placementInputCatalog.dat')


            imageName = '%s_%s_%s.fits' % (imageRoot, det_name, obs.bandpass)

            ra_c, dec_c = raDecFromPixelCoordsLSST(2000.0, 2000.0,
                                                   detector.getName(),
                                                   band=obs.bandpass,
                                                   obs_metadata=obs)

            nSamples = 3
            rng = np.random.RandomState(42)
            fwhm = 0.2

            for iteration in range(nSamples):
                if os.path.exists(dbFileName):
                    os.unlink(dbFileName)

                ra_obj = ra_c + rng.random_sample()*0.2 - 0.1
                dec_obj = dec_c + rng.random_sample()*0.2 - 0.1

                dmx_wrong, dmy_wrong = pixelCoordsFromRaDec(ra_obj, dec_obj,
                                                            chipName=detector.getName(),
                                                            obs_metadata=obs,
                                                            camera=lsst_camera())

                dmx_pix, dmy_pix = pixelCoordsFromRaDecLSST(ra_obj, dec_obj,
                                                            chipName=detector.getName(),
                                                            obs_metadata=obs,
                                                            band=obs.bandpass)

                x_pix, y_pix = pixel_transformer.cameraPixFromDMPix(dmx_pix, dmy_pix,
                                                                    detector.getName())

                x_pix_wrong, y_pix_wrong = pixel_transformer.cameraPixFromDMPix(dmx_wrong, dmy_wrong,
                                                                                detector.getName())

                d_ra = 3600.0*(ra_obj - obs.pointingRA)  # in arcseconds
                d_dec = 3600.0*(dec_obj - obs.pointingDec)

                create_text_catalog(obs, dbFileName,
                                    np.array([d_ra]), np.array([d_dec]),
                                    mag_norm=[magNorm])

                db = LSSTPlacementFileDBObj(dbFileName, runtable='test')
                cat = LSSTPlacementCatalog(db, obs_metadata=obs)
                cat.camera_wrapper = LSSTCameraWrapper()
                psf = SNRdocumentPSF(fwhm=fwhm)
                cat.setPSF(psf)

                cat.write_catalog(catName)
                cat.write_images(nameRoot=imageRoot)

                im = afwImage.ImageF(imageName).getArray()
                tot_flux = im.sum()
                self.assertGreater(tot_flux, 100.0)

                y_centroid = sum([ii*im[ii,:].sum() for ii in range(im.shape[0])])/tot_flux
                x_centroid = sum([ii*im[:,ii].sum() for ii in range(im.shape[1])])/tot_flux
                dd = np.sqrt((x_pix-x_centroid)**2 + (y_pix-y_centroid)**2)
                self.assertLess(dd, 0.5*fwhm)

                dd_wrong = np.sqrt((x_pix_wrong-x_centroid)**2 +
                                   (y_pix_wrong-y_centroid)**2)

                self.assertLess(dd, dd_wrong)

                if os.path.exists(dbFileName):
                    os.unlink(dbFileName)
                if os.path.exists(catName):
                    os.unlink(catName)
                if os.path.exists(imageName):
                    os.unlink(imageName)

        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
