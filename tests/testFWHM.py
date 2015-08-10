import numpy
import os
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.utils import haversine, arcsecFromRadians
from lsst.sims.catalogs.generation.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, GalSimDetector, SNRdocumentPSF
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates

from lsst.sims.coordUtils.utils import ReturnCamera

from  testUtils import get_center_of_detector, create_text_catalog

class fwhmFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    #sedFilename

    columns = [('raJ2000','ra*PI()/180.0', numpy.float),
               ('decJ2000','dec*PI()/180.0', numpy.float),
               ('magNorm', 'mag_norm', numpy.float)]



class fwhmCat(GalSimStars):
    camera = ReturnCamera(os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData'))
    bandpassNames = ['u']

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str,12)),
                        ('properMotionRa', 0.0, numpy.float),
                        ('properMotionDec', 0.0, numpy.float),
                        ('radialVelocity', 0.0, numpy.float),
                        ('parallax', 0.0, numpy.float),
                        ]


class GalSimFwhmTest(unittest.TestCase):

    def verify_fwhm(self, fileName, fwhm, detector, camera, obs, epoch=2000.0):
        im = afwImage.ImageF(fileName).getArray()
        maxFlux = im.max()

        # this looks backwards, but remember: the way numpy handles
        # arrays, the first index indicates what row it is in (the y coordinate)
        _maxPixel = numpy.array([im.argmax()/im.shape[1], im.argmax()%im.shape[1]])
        maxPixel = numpy.array([_maxPixel[1], _maxPixel[0]])

        raMax, decMax = raDecFromPixelCoordinates([maxPixel[0]],
                                                  [maxPixel[1]],
                                                  [detector.getName()],
                                                  camera=camera,
                                                  obs_metadata=obs,
                                                  epoch=epoch)

        half_flux=0.5*maxFlux

        for theta in numpy.arange(0.0, 2.0*numpy.pi, 0.21*numpy.pi):

            slope = numpy.tan(theta)

            if numpy.abs(slope<1.0):
                xPixList = [ix for ix in range(0, im.shape[1]) \
                                if int(slope*(ix-maxPixel[0]) + maxPixel[1])>=0 and int(slope*(ix-maxPixel[0])+maxPixel[1])<im.shape[0]]

                yPixList = [int(slope*(ix-maxPixel[0])+maxPixel[1]) for ix in xPixList]
            else:
                yPixList = [iy for iy in range(0, im.shape[0]) \
                                if int((iy-maxPixel[1])/slope + maxPixel[0])>=0 and int((iy-maxPixel[1])/slope + maxPixel[0])<im.shape[1]]

                xPixList = [int((iy-maxPixel[1])/slope + maxPixel[0]) for iy in yPixList]

            chipNameList = [detector.getName()]*len(xPixList)
            raList, decList = raDecFromPixelCoordinates(xPixList, yPixList, chipNameList,
                                                        camera=camera, obs_metadata=obs, epoch=epoch)

            distanceList = arcsecFromRadians(haversine(raList, decList, raMax[0], decMax[0]))

            fluxList = numpy.array([im[iy][ix] for ix,iy in zip(xPixList, yPixList)])

            distanceToLeft = None
            distanceToRight = None

            for ix in range(1,len(xPixList)):
                if fluxList[ix]<half_flux and fluxList[ix+1]>=half_flux:
                    break

            newOrigin = ix+1

            mm = (distanceList[ix]-distanceList[ix+1])/(fluxList[ix]-fluxList[ix+1])
            bb = distanceList[ix] - mm * fluxList[ix]
            distanceToLeft = mm*half_flux + bb

            for ix in range(newOrigin, len(xPixList)-1):
                if fluxList[ix]>=half_flux and fluxList[ix+1]<half_flux:
                    break

            mm = (distanceList[ix]-distanceList[ix+1])/(fluxList[ix]-fluxList[ix+1])
            bb = distanceList[ix] - mm * fluxList[ix]
            distanceToRight = mm*half_flux + bb

            self.assertTrue(numpy.abs(distanceToLeft+distanceToRight-fwhm) < 0.1*fwhm)


    def testFwhmOfImage(self):
        scratchDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'scratchSpace')
        catName = os.path.join(scratchDir, 'fwhm_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'fwhm_test_Image')
        dbFileName = os.path.join(scratchDir, 'fwhm_test_InputCatalog.dat')

        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(baseDir)

        detector = camera[0]
        detName = detector.getName()
        imageName = '%s_%s_u.fits' % (imageRoot, detName)

        obs = ObservationMetaData(unrefractedRA = 75.0,
                                  unrefractedDec = -12.0,
                                  boundType = 'circle',
                                  boundLength = 4.0,
                                  rotSkyPos = 33.0,
                                  mjd = 49250.0)

        fwhmTestList = [0.5, 0.9, 1.3]

        for fwhm in fwhmTestList:
            create_text_catalog(obs, dbFileName, numpy.array([3.0]), \
                                numpy.array([1.0]), mag_norm=[14.0])

            db = fwhmFileDBObj(dbFileName, runtable='test')

            cat = fwhmCat(db, obs_metadata=obs)
            cat.camera = camera

            psf = SNRdocumentPSF(fwhm=fwhm)
            cat.setPSF(psf)


            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            self.verify_fwhm(imageName, fwhm, detector, camera, obs)

            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(imageName):
                os.unlink(imageName)





def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimFwhmTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
