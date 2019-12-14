import os
import numpy as np
import unittest
import tempfile
import shutil
import lsst.utils.tests

import astropy.io.fits as fits

from lsst.utils import getPackageDir
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from lsst.sims.GalSimInterface import GalSimCameraWrapper
from lsst.sims.GalSimInterface import LSSTCameraWrapper
#from lsst.sims.coordUtils.utils import ReturnCamera

from testUtils import create_text_catalog

from lsst.sims.coordUtils import clean_up_lsst_camera

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class fitsHeaderFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 8123
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float),
               ('magNorm', 'mag_norm', np.float)]


class fitsHeaderCatalog(GalSimStars):

    bandpassNames = ['u']

    def get_galacticRv(self):
        ra = self.column_by_name('raJ2000')
        return np.array([3.1]*len(ra))

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str, 12)),
                        ('properMotionRa', 0.0, np.float),
                        ('properMotionDec', 0.0, np.float),
                        ('radialVelocity', 0.0, np.float),
                        ('parallax', 0.0, np.float)
                        ]

@unittest.skip('ReturnCamera deprecated - need replacement')
class FitsHeaderTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()
        clean_up_lsst_camera()

    def testFitsHeader(self):
        """
        Create a test image with the LSST camera and with the
        cartoon camera.  Verify that the image created with the LSST
        camera has the DM-required cards in its FITS header while the
        image created with the cartoon camera does not
        """

        cameraDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        cartoonCamera = ReturnCamera(cameraDir)

        outputDir = tempfile.mkdtemp(dir=ROOT, prefix='testFitsHeader-')

        lsst_cat_name = os.path.join(outputDir, 'fits_test_lsst_cat.txt')
        lsst_cat_root = os.path.join(outputDir, 'fits_test_lsst_image')

        cartoon_cat_name = os.path.join(outputDir, 'fits_test_cartoon_cat.txt')
        cartoon_cat_root = os.path.join(outputDir, 'fits_test_cartoon_image')

        obs = ObservationMetaData(pointingRA=32.0, pointingDec=22.0,
                                  boundLength=0.1, boundType='circle',
                                  mjd=58000.0, rotSkyPos=14.0, bandpassName='u')

        obs.OpsimMetaData = {'obshistID': 112}

        dbFileName = os.path.join(outputDir, 'fits_test_db.dat')
        create_text_catalog(obs, dbFileName, np.array([30.0]), np.array([30.0]), [22.0])
        db = fitsHeaderFileDBObj(dbFileName, runtable='test')

        # first test the lsst camera
        lsstCat = fitsHeaderCatalog(db, obs_metadata=obs)
        lsstCat.camera_wrapper = LSSTCameraWrapper()
        lsstCat.PSF = SNRdocumentPSF()
        lsstCat.write_catalog(lsst_cat_name)
        lsstCat.write_images(nameRoot=lsst_cat_root)

        list_of_files = os.listdir(outputDir)
        ct = 0
        for file_name in list_of_files:
            true_name = os.path.join(outputDir, file_name)
            if lsst_cat_root in true_name:
                ct += 1
                with fits.open(true_name) as fitsTest:
                    header = fitsTest[0].header
                    self.assertIn('CHIPID', header)
                    self.assertIn('OBSID', header)
                    self.assertIn('OUTFILE', header)
                    self.assertEqual(header['OBSID'], 112)
                    self.assertEqual(header['CHIPID'], 'R22_S11')
                    self.assertEqual(header['OUTFILE'], 'lsst_e_112_f0_R22_S11_E000')
                os.unlink(true_name)

        self.assertGreater(ct, 0)
        if os.path.exists(lsst_cat_name):
            os.unlink(lsst_cat_name)

        # now test with the cartoon camera
        cartoonCat = fitsHeaderCatalog(db, obs_metadata=obs)
        cartoonCat.camera_wrapper = GalSimCameraWrapper(cartoonCamera)
        cartoonCat.PSF = SNRdocumentPSF()
        cartoonCat.write_catalog(cartoon_cat_name)
        cartoonCat.write_images(nameRoot=cartoon_cat_root)
        list_of_files = os.listdir(outputDir)
        ct = 0
        for file_name in list_of_files:
            true_name = os.path.join(outputDir, file_name)
            if cartoon_cat_root in true_name:
                ct += 1
                with fits.open(true_name) as fitsTest:
                    header = fitsTest[0].header
                    self.assertNotIn('CHIPID', header)
                    self.assertNotIn('OBSID', header)
                    self.assertNotIn('OUTFILE', header)
                os.unlink(true_name)

        self.assertGreater(ct, 0)
        if os.path.exists(cartoon_cat_name):
            os.unlink(cartoon_cat_name)

        if os.path.exists(dbFileName):
            os.unlink(dbFileName)

        if os.path.exists(outputDir):
            shutil.rmtree(outputDir)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
