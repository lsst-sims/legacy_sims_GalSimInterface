from builtins import zip
import numpy as np
import os
import unittest
import tempfile
import shutil
import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.geom.angle as afwAngle
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData, arcsecFromRadians
from lsst.sims.utils import haversine
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from lsst.sims.GalSimInterface import GalSimCameraWrapper
from lsst.sims.coordUtils import _raDecFromPixelCoords

from lsst.sims.coordUtils.utils import ReturnCamera

from testUtils import create_text_catalog

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class outputWcsFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    # sedFilename

    columns = [('raJ2000', 'ra*PI()/180.0', np.float),
               ('decJ2000', 'dec*PI()/180.0', np.float)]


class outputWcsCat(GalSimStars):
    bandpassNames = ['u']

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str, 12)),
                        ('properMotionRa', 0.0, np.float),
                        ('properMotionDec', 0.0, np.float),
                        ('radialVelocity', 0.0, np.float),
                        ('parallax', 0.0, np.float),
                        ('magNorm', 14.0, np.float)]


class GalSimOutputWcsTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def testOutputWcsOfImage(self):
        """
        Test that, when GalSim generates an image, in encodes the WCS in a
        way afw can read.  This is done by creating an image,then reading
        it back in, getting its WCS, and comparing the pixel-to-sky conversion
        both for the read WCS and the original afw.cameraGeom.detector.
        Raise an exception if the median difference between the two is
        greater than 0.01 arcseconds.
        """
        scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='testOutputWcsOfImage-')
        catName = os.path.join(scratchDir, 'outputWcs_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'outputWcs_test_Image')
        dbFileName = os.path.join(scratchDir, 'outputWcs_test_InputCatalog.dat')

        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(baseDir)

        detector = camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        nSamples = 3
        rng = np.random.RandomState(42)
        pointingRaList = rng.random_sample(nSamples)*360.0
        pointingDecList = rng.random_sample(nSamples)*180.0 - 90.0
        rotSkyPosList = rng.random_sample(nSamples)*360.0

        for raPointing, decPointing, rotSkyPos in \
            zip(pointingRaList, pointingDecList, rotSkyPosList):

            obs = ObservationMetaData(pointingRA = raPointing,
                                      pointingDec = decPointing,
                                      boundType = 'circle',
                                      boundLength = 4.0,
                                      rotSkyPos = rotSkyPos,
                                      mjd = 49250.0)

            fwhm = 0.7
            create_text_catalog(obs, dbFileName, np.array([3.0]),
                                np.array([1.0]))

            db = outputWcsFileDBObj(dbFileName, runtable='test')

            cat = outputWcsCat(db, obs_metadata=obs)
            cat.camera_wrapper = GalSimCameraWrapper(camera)

            psf = SNRdocumentPSF(fwhm=fwhm)
            cat.setPSF(psf)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            # 20 March 2017
            # the 'try' block is how it worked in SWIG;
            # the 'except' block is how it works in pybind11
            try:
                exposure = afwImage.ExposureD_readFits(imageName)
            except AttributeError:
                exposure = afwImage.ExposureD.readFits(imageName)

            wcs = exposure.getWcs()

            xxTestList = []
            yyTestList = []

            raImage = []
            decImage = []

            for xx in np.arange(0.0, 4001.0, 100.0):
                for yy in np.arange(0.0, 4001.0, 100.0):
                    xxTestList.append(xx)
                    yyTestList.append(yy)

                    pt = afwGeom.Point2D(xx, yy)
                    skyPt = wcs.pixelToSky(pt).getPosition(afwAngle.degrees)
                    raImage.append(skyPt.getX())
                    decImage.append(skyPt.getY())

            xxTestList = np.array(xxTestList)
            yyTestList = np.array(yyTestList)

            raImage = np.radians(np.array(raImage))
            decImage = np.radians(np.array(decImage))

            raControl, \
            decControl = _raDecFromPixelCoords(xxTestList, yyTestList,
                                               [detector.getName()]*len(xxTestList),
                                               camera=camera, obs_metadata=obs,
                                               epoch=2000.0)

            errorList = arcsecFromRadians(haversine(raControl, decControl,
                                                    raImage, decImage))

            medianError = np.median(errorList)
            msg = 'medianError was %e' % medianError
            self.assertLess(medianError, 0.01, msg=msg)

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
