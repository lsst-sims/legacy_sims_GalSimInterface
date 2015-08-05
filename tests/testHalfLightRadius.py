import numpy
import os
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.utils import haversine, arcsecFromRadians
from lsst.sims.catalogs.generation.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimGalaxies, GalSimDetector
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE

from lsst.sims.coordUtils.utils import ReturnCamera

class hlrFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    #sedFilename

    columns = [('raJ2000','ra*PI()/180.0', numpy.float),
               ('decJ2000','dec*PI()/180.0', numpy.float),
               ('halfLightRadius', 'hlr*PI()/648000.0', numpy.float)]



class hlrCat(GalSimGalaxies):
    camera = ReturnCamera(os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData'))
    bandpassNames = ['u']
    default_columns = [('sedFilename', 'sed_flat.txt', (str, 12)),
                       ('magNorm', 21.0, float),
                       ('galacticAv', 0.1, float),
                       ('galSimType', 'sersic', (str,11)),
                       ('internalAv', 0.1, float),
                       ('internalRv', 3.1, float),
                       ('redshift', 0.0, float),
                       ('majorAxis', radiansFromArcsec(1.0), float),
                       ('minorAxis', radiansFromArcsec(1.0), float),
                       ('sindex', 4.0, float),
                       ('positionAngle', 0.0, float)]



class GalSimHlrTest(unittest.TestCase):

    def get_center_of_detector(self, detector, camera, obs, epoch=2000.0):

        pixelSystem = detector.makeCameraSys(PIXELS)
        centerPoint = detector.getCenter(FOCAL_PLANE)
        centerPixel = camera.transform(centerPoint, pixelSystem).getPoint()
        xPix = centerPixel.getX()
        yPix = centerPixel.getY()
        ra, dec = raDecFromPixelCoordinates([xPix], [yPix], [detector.getName()],
                                        camera=camera, obs_metadata=obs, epoch=epoch)

        return ra[0], dec[0]


    def create_text_catalog(self, obs, raCenter, decCenter, hlr, file_name):
        if os.path.exists(file_name):
            os.unlink(file_name)

        dxList = radiansFromArcsec(numpy.array([3.0]))
        dyList = radiansFromArcsec(numpy.array([1.0]))

        raPoint, decPoint = observedFromICRS(numpy.array([obs._unrefractedRA]),
                                             numpy.array([obs._unrefractedDec]),
                                             obs_metadata=obs, epoch=2000.0)

        dx_center = obs._unrefractedRA-raPoint[0]
        dy_center = obs._unrefractedDec-decPoint[0]

        with open(file_name,'w') as outFile:
            outFile.write('# test_id ra dec hlr\n')
            for ix, (dx, dy) in enumerate(zip(dxList, dyList)):
                rr = numpy.degrees(raCenter+dx+dx_center)
                dd = numpy.degrees(decCenter+dy+dy_center)

                outFile.write('%d %.9f %.9f %.9f\n' % (ix, rr, dd, hlr))


    def get_half_light_radius(self, fileName, detector, camera, obs, epoch=2000.0):

        im = afwImage.ImageF(fileName).getArray()
        totalFlux = im.sum()

        maxPixel = numpy.array([im.argmax()/im.shape[1], im.argmax()%im.shape[1]])

        raMax, decMax = raDecFromPixelCoordinates([maxPixel[0]],
                                                  [maxPixel[1]],
                                                  [detector.getName()],
                                                  camera=camera,
                                                  obs_metadata=obs,
                                                  epoch=epoch)

        activePoints = numpy.where(im>1.0e-10)

        xPixList = activePoints[0]
        yPixList = activePoints[1]
        chipNameList = [detector.getName()]*len(xPixList)


        raList, decList = raDecFromPixelCoordinates(xPixList, yPixList, chipNameList,
                                                camera=camera, obs_metadata=obs,
                                                epoch=epoch)

        distanceList = arcsecFromRadians(haversine(raList, decList, raMax[0], decMax[0]))

        for hlr in numpy.arange(0.1, distanceList.max(),0.1):
            dexContained = [ix for ix, dd in enumerate(distanceList) if dd<=hlr]
            countedFlux = numpy.array([im[xPixList[dex]][yPixList[dex]] for dex in dexContained]).sum()
            if countedFlux>=0.5*totalFlux:
                break

        return hlr


    def testHalfLightRadiusOfImage(self):
        catName = 'scratchSpace/hlr_test_Catalog.dat'
        imageRoot = 'scratchSpace/hlr_test_Image'
        dbFileName = 'scratchSpace/hlr_test_InputCatalog.dat'

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

        raCenter, decCenter = self.get_center_of_detector(detector, camera, obs)
        hlrTestList = [1.0, 2.0, 3.0, 4.0]

        for hlr in hlrTestList:
            self.create_text_catalog(obs, raCenter, decCenter, hlr, dbFileName)

            db = hlrFileDBObj(dbFileName, runtable='test')

            cat = hlrCat(db, obs_metadata=obs)
            cat.camera = camera

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            hlrTest = self.get_half_light_radius(imageName, detector, camera, obs)

            self.assertTrue(numpy.abs(hlrTest-hlr)<0.2)


            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(imageName):
                os.unlink(imageName)





def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimHlrTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
