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

from  testUtils import get_center_of_detector

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


    def get_flux_in_half_light_radius(self, fileName, hlr, detector, camera, obs, epoch=2000.0):

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

        dexContained = [ix for ix, dd in enumerate(distanceList) if dd<=hlr]
        countedFlux = numpy.array([im[xPixList[dex]][yPixList[dex]] for dex in dexContained]).sum()
        return totalFlux, countedFlux


    def testHalfLightRadiusOfImage(self):
        scratchDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'scratchSpace')
        catName = os.path.join(scratchDir, 'hlr_test_Catalog.dat')
        imageRoot = os.path.join(scratchDir, 'hlr_test_Image')
        dbFileName = os.path.join(scratchDir, 'hlr_test_InputCatalog.dat')

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

        raCenter, decCenter = get_center_of_detector(detector, camera, obs)
        hlrTestList = [1.0, 2.0, 3.0, 4.0]

        for hlr in hlrTestList:
            self.create_text_catalog(obs, raCenter, decCenter, hlr, dbFileName)

            db = hlrFileDBObj(dbFileName, runtable='test')

            cat = hlrCat(db, obs_metadata=obs)
            cat.camera = camera

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            totalFlux, hlrFlux = self.get_flux_in_half_light_radius(imageName, hlr, detector, camera, obs)
            sigmaFlux = numpy.sqrt(0.5*totalFlux)
            self.assertTrue(numpy.abs(hlrFlux-0.5*totalFlux)<3.0*sigmaFlux)

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
