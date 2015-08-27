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

from lsst.sims.coordUtils.utils import ReturnCamera

from  testUtils import create_text_catalog

class hlrFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    #sedFilename

    columns = [('raJ2000','ra*PI()/180.0', numpy.float),
               ('decJ2000','dec*PI()/180.0', numpy.float),
               ('halfLightRadius', 'hlr*PI()/648000.0', numpy.float),
               ('magNorm', 'mag_norm', numpy.float),
               ('positionAngle', 'pa*PI()/180.0', numpy.float)]



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
                       ('sindex', 4.0, float)]



class GalSimHlrTest(unittest.TestCase):


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

        _maxPixel = numpy.array([im.argmax()/im.shape[1], im.argmax()%im.shape[1]])
        maxPixel = numpy.array([_maxPixel[1], _maxPixel[0]])

        raMax, decMax = raDecFromPixelCoordinates([maxPixel[0]],
                                                  [maxPixel[1]],
                                                  [detector.getName()],
                                                  camera=camera,
                                                  obs_metadata=obs,
                                                  epoch=epoch)

        activePoints = numpy.where(im>1.0e-10)

        xPixList = activePoints[1] # this looks backwards, but remember: the way numpy handles
        yPixList = activePoints[0] # arrays, the first index indicates what row it is in (the y coordinate)
        chipNameList = [detector.getName()]*len(xPixList)


        raList, decList = raDecFromPixelCoordinates(xPixList, yPixList, chipNameList,
                                                camera=camera, obs_metadata=obs,
                                                epoch=epoch)

        distanceList = arcsecFromRadians(haversine(raList, decList, raMax[0], decMax[0]))

        dexContained = [ix for ix, dd in enumerate(distanceList) if dd<=hlr]
        measuredHalfFlux = numpy.array([im[yPixList[dex]][xPixList[dex]] for dex in dexContained]).sum()
        return totalFlux, measuredHalfFlux


    def testHalfLightRadiusOfImage(self):
        """
        Test that GalSim is generating images of objects with the expected half light radius
        by generating images with one object on them and comparing the total flux in the image
        with the flux contained within the expected half light radius.  Raise an exception
        if the deviation is greater than 3-sigma.
        """
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

        hlrTestList = [1.0, 2.0, 3.0, 4.0]

        for hlr in hlrTestList:
            create_text_catalog(obs, dbFileName, numpy.array([3.0]), numpy.array([1.0]),
                                hlr=[hlr])

            db = hlrFileDBObj(dbFileName, runtable='test')

            cat = hlrCat(db, obs_metadata=obs)
            cat.camera = camera

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            totalFlux, hlrFlux = self.get_flux_in_half_light_radius(imageName, hlr, detector, camera, obs)
            sigmaFlux = numpy.sqrt(0.5*totalFlux/cat.photParams.gain) #divide by gain because Poisson stats apply to photons
            self.assertTrue(numpy.abs(hlrFlux-0.5*totalFlux)<4.0*sigmaFlux)

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
