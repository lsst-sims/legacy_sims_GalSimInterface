from builtins import zip
from builtins import range
import numpy as np
import os
import unittest
import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData, arcsecFromRadians
from lsst.sims.utils import angularSeparation
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from lsst.sims.coordUtils import raDecFromPixelCoords

from lsst.sims.coordUtils.utils import ReturnCamera

from testUtils import create_text_catalog


def setup_module(module):
    lsst.utils.tests.init()


class fwhmFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    # sedFilename

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float),
               ('magNorm', 'mag_norm', np.float)]


class fwhmCat(GalSimStars):
    camera = ReturnCamera(os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData'))
    bandpassNames = ['u']

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str, 12)),
                        ('properMotionRa', 0.0, np.float),
                        ('properMotionDec', 0.0, np.float),
                        ('radialVelocity', 0.0, np.float),
                        ('parallax', 0.0, np.float)]


class GalSimFwhmTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def verify_fwhm(self, fileName, fwhm, detector, camera, obs, epoch=2000.0):
        """
        Read in a FITS image with one object on it and verify that that object
        has the expected Full Width at Half Maximum.  This is done by fitting
        the image to the double Gaussian PSF model implemented in
        SNRdocumentPSF(), varying the FWHM.

        @param [in] fileName is the name of the FITS image

        @param [in] fwhm is the expected Full Width at Half Maximum in arcseconds

        @param [in] detector is an instantiation of the afw.cameraGeom Detector
        class characterizing the detector corresponding to this image

        @param [in] camera is an instantiation of the afw.cameraGeom Camera class
        characterizing the camera to which detector belongs

        @param [in] obs is an instantiation of ObservationMetaData characterizing
        the telescope pointing

        @param [in] epoch is the epoch in Julian years of the equinox against which
        RA and Dec are measured.

        This method will raise an exception if the measured Full Width at Half Maximum
        deviates from the expected value by more than ten percent.
        """
        im = afwImage.ImageF(fileName).getArray()
        maxFlux = im.max()
        self.assertGreater(maxFlux, 100.0)  # make sure the image is not blank

        im_flat = im.flatten()
        x_arr = np.array([ii % im.shape[0] for ii in range(len(im_flat))])
        y_arr = np.array([ii // im.shape[0] for ii in range(len(im_flat))])

        valid_pix = np.where(im_flat>1.0e-20)
        im_flat = im_flat[valid_pix].astype(float)
        x_arr = x_arr[valid_pix]
        y_arr = y_arr[valid_pix]

        total_flux = im_flat.sum()

        x_center = (x_arr.astype(float)*im_flat).sum()/total_flux
        y_center = (y_arr.astype(float)*im_flat).sum()/total_flux

        ra_max, dec_max = raDecFromPixelCoords(x_center, y_center, detector.getName(),
                                               camera=camera, obs_metadata=obs,
                                               epoch=epoch)

        ra_p1, dec_p1 = raDecFromPixelCoords(x_center+1, y_center, detector.getName(),
                                             camera=camera, obs_metadata=obs,
                                             epoch=epoch)

        pixel_scale = angularSeparation(ra_max, dec_max, ra_p1, dec_p1)
        pixel_scale *= 3600.0  # convert to arcsec

        chisq_best = None
        fwhm_best = None

        total_flux = im_flat.sum()

        for fwhm_test in np.arange(0.1*fwhm, 3.0*fwhm, 0.1*fwhm):
            alpha = fwhm_test/2.3835

            dd = np.power(x_arr-x_center,2).astype(float) + np.power(y_arr-y_center, 2).astype(float)
            dd *= np.power(pixel_scale, 2)

            sigma = alpha
            g1 = np.exp(-0.5*dd/(sigma*sigma))/(sigma*sigma*2.0*np.pi)

            sigma = 2.0*alpha
            g2 = np.exp(-0.5*dd/(sigma*sigma))/(sigma*sigma*2.0*np.pi)

            model = 0.909*(g1 + 0.1*g2)*pixel_scale*pixel_scale
            norm = model.sum()
            model *= (total_flux/norm)
            chisq = np.power((im_flat-model), 2).sum()

            if chisq_best is None or chisq<chisq_best:
                chisq_best = chisq
                fwhm_best = fwhm_test

        self.assertLess(np.abs(fwhm-fwhm_best), 0.1*fwhm)

    def testFwhmOfImage(self):
        """
        Test that GalSim generates images with the expected Full Width at Half Maximum.
        """
        scratchDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'scratchSpace')
        catName = os.path.join(scratchDir, 'fwhm_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'fwhm_test_Image')
        dbFileName = os.path.join(scratchDir, 'fwhm_test_InputCatalog.dat')

        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(baseDir)

        detector = camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        obs = ObservationMetaData(pointingRA = 75.0,
                                  pointingDec = -12.0,
                                  boundType = 'circle',
                                  boundLength = 4.0,
                                  rotSkyPos = 33.0,
                                  mjd = 49250.0)

        create_text_catalog(obs, dbFileName, np.array([3.0]),
                            np.array([1.0]), mag_norm=[13.0])

        db = fwhmFileDBObj(dbFileName, runtable='test')

        for fwhm in (0.1, 0.14):

            cat = fwhmCat(db, obs_metadata=obs)
            cat.camera = camera

            psf = SNRdocumentPSF(fwhm=fwhm, pixel_scale=0.02)
            cat.setPSF(psf)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            self.verify_fwhm(imageName, fwhm, detector, camera, obs)

            if os.path.exists(catName):
                os.unlink(catName)

            if os.path.exists(imageName):
                os.unlink(imageName)

        if os.path.exists(dbFileName):
            os.unlink(dbFileName)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
