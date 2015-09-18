import unittest
import lsst.utils.tests as utilsTests

import numpy
import os
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians, haversine
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import _pixelCoordsFromRaDec, _raDecFromPixelCoords
from lsst.sims.photUtils import Sed, Bandpass
from lsst.sims.catalogs.generation.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF
from testUtils import create_text_catalog

class placementFileDBObj(fileDBObject):
    idColKey = 'test_id'
    objectTypeId = 88
    tableid = 'test'
    raColName = 'ra'
    decColName = 'dec'
    #sedFilename

    columns = [('raJ2000','ra*PI()/180.0', numpy.float),
               ('decJ2000','dec*PI()/180.0', numpy.float),
               ('magNorm', 'mag_norm', numpy.float)]


class placementCatalog(GalSimStars):

    bandpassNames = ['u']

    def get_galacticRv(self):
        ra = self.column_by_name('raJ2000')
        return numpy.array([3.1]*len(ra))

    default_columns = GalSimStars.default_columns

    default_columns += [('sedFilename', 'sed_flat.txt', (str,12)),
                        ('properMotionRa', 0.0, numpy.float),
                        ('properMotionDec', 0.0, numpy.float),
                        ('radialVelocity', 0.0, numpy.float),
                        ('parallax', 0.0, numpy.float)
                        ]


class GalSimPlacementTest(unittest.TestCase):

    def setUp(self):
        self.magNorm=20.0

    def check_placement(self, imageName, raList, decList, fwhmList,
                        countList, gain,
                        detector, camera, obs, epoch=2000.0):
        """
        Read in a FITS image and a list of objects meant to be on that
        image.  Verify that the objects were placed at the correct pixel
        by counting up all of the flux within 2 fwhm of each object's
        expected location and verifying it with the counts expected for
        that object.

        @param [in] imageName is the name of the FITS file to be read in

        @param [in] raList is a numpy array of the RA coordinates of the objects
        in the image (in radians)

        @param [in] decList is a numpy array of the Dec coordinates of the objects
        in the image (in radians)

        @param [in] fwhmList is a list of the Full Width at Half Maximum of
        each object in arcseconds

        @param [in] countList is a list of the counts expected for each object

        @param [in] gain is the gain of the detector (electrons per ADU)

        @param [in] detector is an instantiation of the afw.cameraGeom Detector
        class characterizing the detector corresponding to this image

        @param [in] camera is an instantiation of the afw.cameraGeom Camera class
        characterizing the camera to which detector belongs

        @param [in] obs is an instantiation of ObservationMetaData characterizing
        the telescope pointing

        @param [in] epoch is the epoch in Julian years of the equinox against which
        RA and Dec are measured.

        Raises an exception of the counts detected for each object differs from
        the expected amount by more than 3 sigma.
        """


        im = afwImage.ImageF(imageName).getArray()
        activePixels = numpy.where(im>1.0e-10)

        # I know this seems backwards, but the way numpy handles arrays,
        # the first index is the row (i.e. the y coordinate)
        imXList = activePixels[1]
        imYList = activePixels[0]

        nameList = [detector.getName()]*len(raList)
        xPixList, yPixList = _pixelCoordsFromRaDec(raList, decList,
                                                   chipNames=nameList,
                                                   camera=camera,
                                                   obs_metadata=obs,
                                                   epoch=epoch)

        for rr, dd, xx, yy, fwhm, cc in \
        zip(raList, decList, xPixList, yPixList, fwhmList, countList):
            countSigma = numpy.sqrt(cc/gain)

            imNameList = [detector.getName()]*len(imXList)
            raImList, decImList = _raDecFromPixelCoords(imXList, imYList,
                                                        imNameList,
                                                        camera=camera,
                                                        obs_metadata=obs,
                                                        epoch=epoch)

            distanceList = arcsecFromRadians(haversine(raImList, decImList, rr, dd))

            fluxArray = numpy.array(
                                  [im[imYList[ix]][imXList[ix]] \
                                   for ix in range(len(distanceList)) \
                                   if distanceList[ix]<2.0*fwhm]
                                  )

            totalFlux = fluxArray.sum()
            self.assertTrue(numpy.abs(totalFlux-cc)<3.0*countSigma)



    def testObjectPlacement(self):
        """
        Test that GalSim places objects on the correct pixel by drawing
        images, reading them in, and then comparing the flux contained in
        circles of 2 fwhm radii about the object's expected positions with
        the actual expected flux of the objects.
        """
        scratchDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'scratchSpace')
        catName = os.path.join(scratchDir, 'placementCatalog.dat')
        imageRoot = os.path.join(scratchDir, 'placementImage')
        dbFileName = os.path.join(scratchDir, 'placementInputCatalog.dat')

        cameraDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        camera = ReturnCamera(cameraDir)
        detector = camera[0]
        imageName = '%s_%s_u.fits' % (imageRoot, detector.getName())

        controlSed = Sed()
        controlSed.readSED_flambda(
                                   os.path.join(getPackageDir('sims_sed_library'),
                                               'flatSED','sed_flat.txt.gz')
                                   )

        uBandpass = Bandpass()
        uBandpass.readThroughput(
                                 os.path.join(getPackageDir('throughputs'),
                                              'baseline','total_u.dat')
                                )

        controlBandpass = Bandpass()
        controlBandpass.imsimBandpass()

        ff = controlSed.calcFluxNorm(self.magNorm, uBandpass)
        controlSed.multiplyFluxNorm(ff)
        a_int, b_int = controlSed.setupCCMab()
        controlSed.addCCMDust(a_int, b_int, A_v=0.1, R_v=3.1)

        nSamples = 5
        numpy.random.seed(42)
        pointingRaList = numpy.random.random_sample(nSamples)*360.0
        pointingDecList = numpy.random.random_sample(nSamples)*180.0 - 90.0
        rotSkyPosList = numpy.random.random_sample(nSamples)*360.0
        fwhmList = numpy.random.random_sample(nSamples)*1.0 + 0.3

        actualCounts = None

        for pointingRA, pointingDec, rotSkyPos, fwhm in \
        zip(pointingRaList, pointingDecList, rotSkyPosList, fwhmList):


            obs = ObservationMetaData(unrefractedRA=pointingRA,
                                      unrefractedDec=pointingDec,
                                      boundType='circle',
                                      boundLength=4.0,
                                      mjd=49250.0,
                                      rotSkyPos=rotSkyPos)

            xDisplacementList = numpy.random.random_sample(nSamples)*60.0-30.0
            yDisplacementList = numpy.random.random_sample(nSamples)*60.0-30.0
            create_text_catalog(obs, dbFileName, xDisplacementList, yDisplacementList,
                                mag_norm=[self.magNorm]*len(xDisplacementList))
            db = placementFileDBObj(dbFileName, runtable='test')
            cat = placementCatalog(db, obs_metadata=obs)
            if actualCounts is None:
                actualCounts = controlSed.calcADU(uBandpass, cat.photParams)

            psf = SNRdocumentPSF(fwhm=fwhm)
            cat.setPSF(psf)
            cat.camera = camera

            cat.write_catalog(catName)
            cat.write_images(nameRoot=imageRoot)

            objRaList = []
            objDecList = []
            with open(catName, 'r') as inFile:
                for line in inFile:
                    if line[0] != '#':
                        words = line.split(';')
                        objRaList.append(numpy.radians(numpy.float(words[2])))
                        objDecList.append(numpy.radians(numpy.float(words[3])))

            objRaList = numpy.array(objRaList)
            objDecList = numpy.array(objDecList)

            self.check_placement(imageName, objRaList, objDecList,
                                [fwhm]*len(objRaList),
                                numpy.array([actualCounts]*len(objRaList)),
                                cat.photParams.gain, detector, camera, obs, epoch=2000.0)

            if os.path.exists(dbFileName):
                os.unlink(dbFileName)
            if os.path.exists(catName):
                os.unlink(catName)
            if os.path.exists(imageName):
                os.unlink(imageName)


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimPlacementTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
