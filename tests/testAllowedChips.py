import unittest
import numpy as np
import os
import tempfile
import shutil
import lsst.utils.tests

from lsst.utils import getPackageDir

import lsst.afw.cameraGeom.testUtils as camTestUtils

import lsst.afw.image as afwImage
from lsst.afw.cameraGeom import DetectorType
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.coordUtils import raDecFromPixelCoords
from lsst.sims.photUtils import Sed, Bandpass, BandpassDict, PhotometricParameters
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from lsst.sims.GalSimInterface import GalSimCameraWrapper
from testUtils import create_text_catalog

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class allowedChipsFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float),
               ('magNorm', 'mag_norm', np.float)]


class allowedChipsCatalog(GalSimStars):

    bandpassNames = ['u']

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


class allowedChipsTest(unittest.TestCase):

    longMessage = True

    @classmethod
    def setUpClass(cls):
        cls.scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='allowedChipsTest-')
        cls.obs = ObservationMetaData(pointingRA=122.0, pointingDec=-29.1,
                                      mjd=57381.2, rotSkyPos=43.2,
                                      bandpassName='r')

        cls.camera = camTestUtils.CameraWrapper().camera

        cls.dbFileName = os.path.join(cls.scratchDir, 'allowed_chips_test_db.txt')
        if os.path.exists(cls.dbFileName):
            os.unlink(cls.dbFileName)

        cls.controlSed = Sed()
        cls.controlSed.readSED_flambda(os.path.join(getPackageDir('sims_sed_library'),
                                                    'flatSED', 'sed_flat.txt.gz'))
        cls.magNorm = 18.1
        imsim = Bandpass()
        imsim.imsimBandpass()
        ff = cls.controlSed.calcFluxNorm(cls.magNorm, imsim)
        cls.controlSed.multiplyFluxNorm(ff)
        a_x, b_x = cls.controlSed.setupCCM_ab()
        cls.controlSed.addDust(a_x, b_x, A_v=0.1, R_v=3.1)
        bpd = BandpassDict.loadTotalBandpassesFromFiles()
        pp = PhotometricParameters()
        cls.controlADU = cls.controlSed.calcADU(bpd['u'], pp)
        cls.countSigma = np.sqrt(cls.controlADU/pp.gain)

        cls.x_pix = 50
        cls.y_pix = 50

        x_list = []
        y_list = []
        name_list = []
        for dd in cls.camera:
            x_list.append(cls.x_pix)
            y_list.append(cls.y_pix)
            name_list.append(dd.getName())

        x_list = np.array(x_list)
        y_list = np.array(y_list)

        ra_list, dec_list = raDecFromPixelCoords(x_list, y_list, name_list,
                                                 camera=cls.camera, obs_metadata=cls.obs,
                                                 epoch=2000.0)

        dra_list = 3600.0*(ra_list-cls.obs.pointingRA)
        ddec_list = 3600.0*(dec_list-cls.obs.pointingDec)

        create_text_catalog(cls.obs, cls.dbFileName, dra_list, ddec_list,
                            mag_norm=[cls.magNorm]*len(dra_list))

        cls.db = allowedChipsFileDBObj(cls.dbFileName, runtable='test')

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()
        del cls.camera
        if os.path.exists(cls.dbFileName):
            os.unlink(cls.dbFileName)
        if os.path.exists(cls.scratchDir):
            shutil.rmtree(cls.scratchDir)

    def testCamera(self):
        """
        Test that GalSimCatalogs respect the allowed_chips variable by
        generating a catalog with one object on each chip.

        Generate images from a control catalog that allows all chips.
        Verify that each image contains the expected flux.

        Generate images from a test catalog that only allows two chips.
        Verify that only the two expected images exist and that each
        contains only the expected flux.
        """

        controlCatalog = allowedChipsCatalog(self.db, obs_metadata=self.obs)
        testCatalog = allowedChipsCatalog(self.db, obs_metadata=self.obs)
        psf = SNRdocumentPSF()
        controlCatalog.setPSF(psf)
        testCatalog.setPSF(psf)
        controlCatalog.camera_wrapper = GalSimCameraWrapper(self.camera)
        testCatalog.camera_wrapper = GalSimCameraWrapper(self.camera)

        test_root = os.path.join(self.scratchDir, 'allowed_chip_test_image')
        control_root = os.path.join(self.scratchDir, 'allowed_chip_control_image')

        name_list = []
        for dd in self.camera:
            if dd.getType() == DetectorType.WAVEFRONT or dd.getType() == DetectorType.GUIDER:
                continue
            name = dd.getName()
            name_list.append(name)
            stripped_name = name.replace(':', '')
            stripped_name = stripped_name.replace(',', '')
            stripped_name = stripped_name.replace(' ', '_')

            test_image_name = os.path.join(self.scratchDir, test_root+'_'+stripped_name+'_u.fits')
            control_image_name = os.path.join(self.scratchDir, control_root+'_'+stripped_name+'_u.fits')

            # remove any images that were generated the last time this test
            # was run
            if os.path.exists(test_image_name):
                os.unlink(test_image_name)
            if os.path.exists(control_image_name):
                os.unlink(control_image_name)

        # only allow two chips in the test catalog
        allowed_chips = [name_list[3], name_list[4]]

        testCatalog.allowed_chips = allowed_chips

        test_cat_name = os.path.join(self.scratchDir, 'allowed_chips_test_cat.txt')
        control_cat_name = os.path.join(self.scratchDir, 'allowed_chips_control_cat.txt')

        testCatalog.write_catalog(test_cat_name)
        controlCatalog.write_catalog(control_cat_name)

        testCatalog.write_images(nameRoot=test_root)
        controlCatalog.write_images(nameRoot=control_root)

        test_image_ct = 0

        for name in name_list:
            # Loop through each chip on the camera.
            # Verify that the control catalog generated an image for each chip.
            # Verify that the test catalog only generated images for the two
            # specified chips.
            # Verify that each image contains the expected amount of flux.

            stripped_name = name.replace(':', '')
            stripped_name = stripped_name.replace(',', '')
            stripped_name = stripped_name.replace(' ', '_')

            test_image_name = os.path.join(self.scratchDir, test_root+'_'+stripped_name+'_u.fits')
            control_image_name = os.path.join(self.scratchDir, control_root+'_'+stripped_name+'_u.fits')

            msg = '%s does not exist; it should' % control_image_name
            self.assertTrue(os.path.exists(control_image_name), msg=msg)
            im = afwImage.ImageF(control_image_name).getArray()
            msg="\nimage contains %e counts\nshould contain %e\n\n" % (im.sum(), self.controlADU)
            self.assertLess(np.abs(im.sum()-self.controlADU), 3.0*self.countSigma,
                            msg=msg)
            os.unlink(control_image_name)

            if name in allowed_chips:
                msg = '%s does not exist; it should' % test_image_name
                self.assertTrue(os.path.exists(test_image_name), msg=msg)
                im = afwImage.ImageF(test_image_name).getArray()
                self.assertLess(np.abs(im.sum()-self.controlADU), 3.0*self.countSigma)
                os.unlink(test_image_name)
                test_image_ct += 1
            else:
                msg = '%s exists; it should not' % test_image_name
                self.assertFalse(os.path.exists(test_image_name), msg=msg)

        self.assertEqual(test_image_ct, len(allowed_chips))

        if os.path.exists(test_cat_name):
            os.unlink(test_cat_name)
        if os.path.exists(control_cat_name):
            os.unlink(control_cat_name)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
