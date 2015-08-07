import unittest
import lsst.utils.tests as utilsTests

import numpy
import os
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians, haversine
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import calculatePixelCoordinates, raDecFromPixelCoordinates
from lsst.sims.photUtils import Sed, Bandpass
from lsst.sims.catalogs.generation.db import fileDBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF

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
        self.magNorm = 20.0

    def create_text_catalog(self, obs, file_name, xDisplacement, yDisplacement):

        if os.path.exists(file_name):
            os.unlink(file_name)

        dxList = radiansFromArcsec(xDisplacement)
        dyList = radiansFromArcsec(yDisplacement)


        with open(file_name,'w') as outFile:
            outFile.write('# test_id ra dec mag_norm\n')
            for ix, (dx, dy) in enumerate(zip(dxList, dyList)):

                rr = numpy.degrees(obs._unrefractedRA+dx)
                dd = numpy.degrees(obs._unrefractedDec+dy)

                outFile.write('%d %.9f %.9f %f\n' % (ix, rr, dd, self.magNorm))


    def check_placement(self, imageName, raList, decList, fwhmList,
                        countList, gain,
                        detector, camera, obs, epoch=2000.0):

        im = afwImage.ImageF(imageName).getArray()
        activePixels = numpy.where(im>1.0e-10)

        # I know this seems backwards, but the way numpy handles arrays,
        # the first index is the row (i.e. the y coordinate)
        imXList = activePixels[1]
        imYList = activePixels[0]

        nameList = [detector.getName()]*len(raList)
        xPixList, yPixList = calculatePixelCoordinates(ra=raList, dec=decList,
                                                       chipNames=nameList,
                                                       camera=camera,
                                                       obs_metadata=obs,
                                                       epoch=epoch)

        for rr, dd, xx, yy, fwhm, cc in \
        zip(raList, decList, xPixList, yPixList, fwhmList, countList):
            countSigma = numpy.sqrt(cc/gain)

            imNameList = [detector.getName()]*len(imXList)
            raImList, decImList = raDecFromPixelCoordinates(imXList, imYList,
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
            print totalFlux, cc, numpy.abs(totalFlux-cc)/countSigma
            self.assertTrue(numpy.abs(totalFlux-cc)<3.0*countSigma)



    def testObjectPlacement(self):
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
            self.create_text_catalog(obs, dbFileName, xDisplacementList, yDisplacementList)
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
