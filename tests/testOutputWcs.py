import numpy
import os
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.utils import haversine, arcsecFromRadians
from lsst.sims.catalogs.generation.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, GalSimDetector, SNRdocumentPSF
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates

from lsst.sims.coordUtils.utils import ReturnCamera

from  testUtils import get_center_of_detector, create_text_catalog

class outputWcsFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    #sedFilename

    columns = [('raJ2000','ra*PI()/180.0', numpy.float),
               ('decJ2000','dec*PI()/180.0', numpy.float)]



class outputWcsCat(GalSimStars):
    camera = ReturnCamera(os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData'))
    bandpassNames = ['u']

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str,12)),
                        ('properMotionRa', 0.0, numpy.float),
                        ('properMotionDec', 0.0, numpy.float),
                        ('radialVelocity', 0.0, numpy.float),
                        ('parallax', 0.0, numpy.float),
                        ('magNorm', 14.0, numpy.float)
                        ]


class GalSimOutputWcsTest(unittest.TestCase):

    def testOutputWcsOfImage(self):
        scratchDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'scratchSpace')
        catName = os.path.join(scratchDir, 'outputWcs_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'outputWcs_test_Image')
        dbFileName = os.path.join(scratchDir, 'outputWcs_test_InputCatalog.dat')

        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(baseDir)

        detector = camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        nSamples = 5
        numpy.random.seed(42)
        pointingRaList = numpy.random.random_sample(nSamples)*360.0
        pointingDecList = numpy.random.random_sample(nSamples)*180.0 - 90.0
        rotSkyPosList = numpy.random.random_sample(nSamples)*360.0

        for raPointing, decPointing, rotSkyPos in \
        zip(pointingRaList, pointingDecList, rotSkyPosList):

            obs = ObservationMetaData(unrefractedRA = raPointing,
                                      unrefractedDec = decPointing,
                                      boundType = 'circle',
                                      boundLength = 4.0,
                                      rotSkyPos = rotSkyPos,
                                      mjd = 49250.0)

            fwhm = 0.7
            create_text_catalog(obs, dbFileName, numpy.array([3.0]),
                                numpy.array([1.0]))

            db = outputWcsFileDBObj(dbFileName, runtable='test')

            cat = outputWcsCat(db, obs_metadata=obs)
            cat.camera = camera

            psf = SNRdocumentPSF(fwhm=fwhm)
            cat.setPSF(psf)

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            exposure = afwImage.ExposureD_readFits(imageName)
            wcs  = exposure.getWcs()

            xxTestList = []
            yyTestList = []

            raImage = []
            decImage = []

            for xx in numpy.arange(0.0, 4001.0, 100.0):
                for yy in numpy.arange(0.0, 4001.0, 100.0):
                    xxTestList.append(xx)
                    yyTestList.append(yy)

                    pt = afwGeom.Point2D(xx ,yy)
                    skyPt = wcs.pixelToSky(pt).getPosition()
                    raImage.append(skyPt.getX())
                    decImage.append(skyPt.getY())

            xxTestList = numpy.array(xxTestList)
            yyTestList = numpy.array(yyTestList)

            raImage = numpy.radians(numpy.array(raImage))
            decImage = numpy.radians(numpy.array(decImage))

            raControl, \
            decControl = raDecFromPixelCoordinates(
                                                   xxTestList, yyTestList,
                                                   [detector.getName()]*len(xxTestList),
                                                   camera=camera, obs_metadata=obs,
                                                   epoch=2000.0
                                                  )

            errorList = arcsecFromRadians(haversine(raControl, decControl,
                                                    raImage, decImage))


            medianError = numpy.median(errorList)
            msg = 'medianError was %e' % medianError
            self.assertTrue(medianError<0.01, msg=msg)

            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(imageName):
                os.unlink(imageName)





def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimOutputWcsTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
