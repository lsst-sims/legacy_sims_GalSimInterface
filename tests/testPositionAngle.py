from builtins import zip
import numpy as np
import os
import unittest
import tempfile
import shutil
import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils.CodeUtilities import sims_clean_up
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec
from lsst.sims.catalogs.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimGalaxies
from lsst.sims.GalSimInterface import GalSimCameraWrapper
from lsst.sims.coordUtils import _raDecFromPixelCoords

from lsst.sims.coordUtils.utils import ReturnCamera

from testUtils import create_text_catalog

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class paFileDBObj(fileDBObject):
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


class paCat(GalSimGalaxies):
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
                       ('minorAxis', radiansFromArcsec(0.5), float),
                       ('sindex', 4.0, float),
                       ('npoints', 0, int),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float),
                       ]


class GalSimPositionAngleTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def get_position_angle(self, imageName, afwCamera, afwDetector,
                           obs_metadata, epoch):
        """
        Read in a FITS image containing one extended object.

        Determine its north and east axes by examining how RA and Dec change
        with pixel position.

        Determine the semi-major axis of the object by treating the distribution
        of flux as a covariance matrix and finding its eigen vectors.

        Return the angle between the semi-major axis and the north axis of
        the image

        @param [in] imageName is the name of the FITS image to be read

        @param [in] afwCamera is an afw.cameraGeom.Camera

        @param [in] afwDetector is an afw.cameraGeom.Detector

        @param [in] obs_metadata is an ObservationMetaData describing the
        pointing of the telescope

        @param [in] epoch is the epoch in Julian years of the equinox against
        which RA and Dec are measured

        @param [out] the position angle of the object in the image in degrees
        """

        im = afwImage.ImageF(imageName).getArray()
        activePixels = np.where(im > 1.0e-10)
        self.assertGreater(len(activePixels), 0)
        xPixList = activePixels[1]
        yPixList = activePixels[0]

        xCenterPix = np.array([im.shape[1]/2])
        yCenterPix = np.array([im.shape[0]/2])

        raCenter, decCenter = _raDecFromPixelCoords(xCenterPix, yCenterPix,
                                                    [afwDetector.getName()],
                                                    camera=afwCamera,
                                                    obs_metadata=obs_metadata,
                                                    epoch=epoch)

        xCenterP1 = xCenterPix+1
        yCenterP1 = yCenterPix+1
        raCenterP1, decCenterP1 = _raDecFromPixelCoords(xCenterP1, yCenterP1,
                                                        [afwDetector.getName()],
                                                        camera=afwCamera,
                                                        obs_metadata=obs_metadata,
                                                        epoch=epoch)

        # find the angle between the (1,1) vector in pixel space and the
        # north axis of the image
        theta = np.arctan2((raCenterP1[0]-raCenter[0]), decCenterP1[0]-decCenter[0])

        # rotate the (1,1) vector in pixel space so that it is pointing
        # along the north axis
        north = np.array([np.cos(theta)-np.sin(theta), np.cos(theta)+np.sin(theta)])
        north = north/np.sqrt(north[0]*north[0]+north[1]*north[1])

        # find the east axis of the image
        east = np.array([north[1], -1.0*north[0]])

        # now find the covariance matrix of the x, y  pixel space distribution
        # of flux on the image
        maxPixel = np.array([im.argmax()%im.shape[1], im.argmax()/im.shape[1]])

        xx = np.array([im[iy][ix]*np.power(ix-maxPixel[0], 2)
                       for ix, iy in zip(xPixList, yPixList)]).sum()

        xy = np.array([im[iy][ix]*(ix-maxPixel[0])*(iy-maxPixel[1])
                       for ix, iy in zip(xPixList, yPixList)]).sum()

        yy = np.array([im[iy][ix]*(iy-maxPixel[1])*(iy-maxPixel[1])
                       for ix, iy in zip(xPixList, yPixList)]).sum()

        covar = np.array([[xx, xy], [xy, yy]])

        # find the eigen vectors of this covarinace matrix;
        # treat the one with the largest eigen value as the
        # semi-major axis of the object
        eigenVals, eigenVecs = np.linalg.eig(covar)

        iMax = eigenVals.argmax()
        majorAxis = eigenVecs[:, iMax]

        majorAxis = majorAxis/np.sqrt(majorAxis[0]*majorAxis[0]+majorAxis[1]*majorAxis[1])

        # return the angle between the north axis of the image
        # and the semi-major axis of the object
        cosTheta = np.dot(majorAxis, north)
        sinTheta = np.dot(majorAxis, east)

        theta = np.arctan2(sinTheta, cosTheta)

        return np.degrees(theta)

    def testPositionAngle(self):
        """
        Test that GalSim is generating images with the correct position angle
        by creating a FITS image with one extended source in it.  Measure
        the angle between the semi-major axis of the source and the north
        axis of the image.  Throw an exception if that angle differs
        from the expected position angle by more than 2 degrees.
        """
        scratchDir = tempfile.mkdtemp(dir=ROOT, prefix='testPositionAngle-')
        catName = os.path.join(scratchDir, 'pa_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'pa_test_Image')
        dbFileName = os.path.join(scratchDir, 'pa_test_InputCatalog.dat')

        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(baseDir)
        detector = camera[0]
        detName = detector.getName()

        rng = np.random.RandomState(42)
        paList = rng.random_sample(7)*360.0
        rotSkyPosList = rng.random_sample(77)*360.0

        for pa, rotSkyPos in zip(paList, rotSkyPosList):

            imageName = '%s_%s_u.fits' % (imageRoot, detName)

            obs = ObservationMetaData(pointingRA = 75.0,
                                      pointingDec = -12.0,
                                      boundType = 'circle',
                                      boundLength = 4.0,
                                      rotSkyPos = rotSkyPos,
                                      mjd = 49250.0)

            create_text_catalog(obs, dbFileName,
                                rng.random_sample(1)*20.0-10.0,
                                rng.random_sample(1)*20.0-10.0,
                                pa=[pa],
                                mag_norm=[17.0])

            db = paFileDBObj(dbFileName, runtable='test')

            cat = paCat(db, obs_metadata=obs)
            cat.camera_wrapper = GalSimCameraWrapper(camera)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            paTest = self.get_position_angle(imageName, camera, detector, obs, 2000.0)

            # need to compare against all angles displaced by either 180 or 360 degrees
            # from expected answer
            deviation = np.abs(np.array([pa-paTest,
                                         pa-180.0-paTest,
                                         pa+180.0-paTest,
                                         pa-360.0-paTest,
                                         pa+360.0-paTest])).min()

            self.assertLess(deviation, 3.0)

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
