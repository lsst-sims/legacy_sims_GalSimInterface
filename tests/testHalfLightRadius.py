import numpy as np
import os
import unittest
import tempfile
import shutil
import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.utils import haversine
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimGalaxies, GalSimRandomWalk
from lsst.sims.GalSimInterface import GalSimCameraWrapper
from lsst.sims.coordUtils import _raDecFromPixelCoords

#from lsst.sims.coordUtils.utils import ReturnCamera

from testUtils import create_text_catalog

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class hlrFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    # sedFilename

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float),
               ('halfLightRadius', 'hlr*PI()/648000.0', np.float),
               ('magNorm', 'mag_norm', np.float),
               ('positionAngle', 'pa*PI()/180.0', np.float)]


class hlrCatSersic(GalSimGalaxies):
    bandpassNames = ['u']
    default_columns = [('sedFilename', 'sed_flat.txt', (str, 12)),
                       ('magNorm', 21.0, float),
                       ('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'sersic', (str, 11)),
                       ('internalAv', 0.1, float),
                       ('internalRv', 3.1, float),
                       ('redshift', 0.0, float),
                       ('majorAxis', radiansFromArcsec(1.0), float),
                       ('minorAxis', radiansFromArcsec(1.0), float),
                       ('sindex', 4.0, float),
                       ('npoints', 0, int),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float),
                       ]

class hlrCatRandomWalk(GalSimRandomWalk):
    bandpassNames = ['u']
    default_columns = [('sedFilename', 'sed_flat.txt', (str, 12)),
                       ('magNorm', 21.0, float),
                       ('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'RandomWalk', (str, 10)),
                       ('internalAv', 0.1, float),
                       ('internalRv', 3.1, float),
                       ('redshift', 0.0, float),
                       ('majorAxis', radiansFromArcsec(1.0), float),
                       ('minorAxis', radiansFromArcsec(1.0), float),
                       ('npoints', 10000, int),
                       ('sindex', 0.0, float),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float),
                       ]
@unittest.skip('ReturnCamera deprecated - need replacement')
class GalSimHlrTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.camera = ReturnCamera(os.path.join(getPackageDir('sims_GalSimInterface'),
                                  'tests', 'cameraData'))

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()
        del cls.camera

    def get_flux_in_half_light_radius(self, fileName, hlr, detector, camera, obs, epoch=2000.0):
        """
        Read in a FITS image.  Return the total flux in that image as well as the flux contained
        within a specified radius of the maximum pixel of the image.

        @param [in] fileName is the name of the FITS file to be read in

        @param [in] hlr is the half light radius to be tested (in arc seconds)

        @param [in] detector is an instantiation of the afw.cameraGeom Detector
        class characterizing the detector corresponding to this image

        @param [in] camera is an instantiation of the afw.cameraGeom Camera class
        characterizing the camera to which detector belongs

        @param [in] obs is an instantiation of ObservationMetaData characterizing
        the telescope pointing

        @param [in] epoch is the epoch in Julian years of the equinox against which
        RA and Dec are measured.

        @param [out] totalFlux is the total number of counts in the images

        @param [out] measuredHalfFlux is the measured flux within hlr of the maximum pixel
        """

        im = afwImage.ImageF(fileName).getArray()
        totalFlux = im.sum()

        activePoints = np.where(im > 1.0e-10)
        self.assertGreater(len(activePoints), 0)

        xPixList = activePoints[1]  # this looks backwards, but remember: the way numpy handles
        yPixList = activePoints[0]  # arrays, the first index indicates what row it is in (the y coordinate)
        chipNameList = [detector.getName()]*len(xPixList)

        # Compute luminosity weighted centroid
        xCen = np.sum(im[yPixList, xPixList] * xPixList) / np.sum(im[yPixList, xPixList])
        yCen = np.sum(im[yPixList, xPixList] * yPixList) / np.sum(im[yPixList, xPixList])
        cenPixel = np.array([xCen, yCen])

        raCen, decCen = _raDecFromPixelCoords(cenPixel[0:1],
                                              cenPixel[1:2],
                                              [detector.getName()],
                                              camera=camera,
                                              obs_metadata=obs,
                                              epoch=epoch)

        raList, decList = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                                camera=camera, obs_metadata=obs,
                                                epoch=epoch)

        distanceList = arcsecFromRadians(haversine(raList, decList, raCen[0], decCen[0]))

        dexContained = [ix for ix, dd in enumerate(distanceList) if dd <= hlr]
        measuredHalfFlux = np.array([im[yPixList[dex]][xPixList[dex]] for dex in dexContained]).sum()
        return totalFlux, measuredHalfFlux

    def testHalfLightRadiusOfImageSersic(self):
        """
        Test that GalSim is generating images of objects with the expected half light radius
        by generating images with one object on them and comparing the total flux in the image
        with the flux contained within the expected half light radius.  Raise an exception
        if the deviation is greater than 3-sigma.
        """
        scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='testHalfLightRadiusOfImage-')
        catName = os.path.join(scratchDir, 'hlr_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'hlr_test_Image')
        dbFileName = os.path.join(scratchDir, 'hlr_test_InputCatalog.dat')

        detector = self.camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        obs = ObservationMetaData(pointingRA = 75.0,
                                  pointingDec = -12.0,
                                  boundType = 'circle',
                                  boundLength = 4.0,
                                  rotSkyPos = 33.0,
                                  mjd = 49250.0)

        hlrTestList = [1.0, 2.0, 4.0]

        for hlr in hlrTestList:
            create_text_catalog(obs, dbFileName, np.array([3.0]), np.array([1.0]),
                                hlr=[hlr])

            db = hlrFileDBObj(dbFileName, runtable='test')

            cat = hlrCatSersic(db, obs_metadata=obs)
            cat.camera_wrapper = GalSimCameraWrapper(self.camera)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            totalFlux, hlrFlux = self.get_flux_in_half_light_radius(imageName, hlr, detector, self.camera, obs)
            self.assertGreater(totalFlux, 1000.0)  # make sure the image is not blank

            # divide by gain because Poisson stats apply to photons
            sigmaFlux = np.sqrt(0.5*totalFlux/cat.photParams.gain)
            self.assertLess(np.abs(hlrFlux-0.5*totalFlux), 4.0*sigmaFlux)

            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(imageName):
                os.unlink(imageName)

        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)

    def testHalfLightRadiusOfImageRandomWalk(self):
        """
        Test that GalSim is generating images of objects with the expected half light radius
        by generating images with one object on them and comparing the total flux in the image
        with the flux contained within the expected half light radius.  Raise an exception
        if the deviation is greater than 3-sigma.
        """
        scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='testHalfLightRadiusOfImage-')
        catName = os.path.join(scratchDir, 'hlr_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'hlr_test_Image')
        dbFileName = os.path.join(scratchDir, 'hlr_test_InputCatalog.dat')

        detector = self.camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        obs = ObservationMetaData(pointingRA = 75.0,
                                  pointingDec = -12.0,
                                  boundType = 'circle',
                                  boundLength = 4.0,
                                  rotSkyPos = 33.0,
                                  mjd = 49250.0)

        hlrTestList = [1., 2., 4.]

        for hlr in hlrTestList:
            create_text_catalog(obs, dbFileName, np.array([3.0]), np.array([1.0]),
                                hlr=[hlr])

            db = hlrFileDBObj(dbFileName, runtable='test')

            cat = hlrCatRandomWalk(db, obs_metadata=obs)
            cat.camera_wrapper = GalSimCameraWrapper(self.camera)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            totalFlux, hlrFlux = self.get_flux_in_half_light_radius(imageName, hlr, detector, self.camera, obs)
            self.assertGreater(totalFlux, 1000.0)  # make sure the image is not blank
            
            # divide by gain because Poisson stats apply to photons
            sigmaFlux = np.sqrt(0.5*totalFlux/cat.photParams.gain)
            self.assertLess(np.abs(hlrFlux-0.5*totalFlux), 4.0*sigmaFlux)

            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(imageName):
                os.unlink(imageName)

        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)

class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
