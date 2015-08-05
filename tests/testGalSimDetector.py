import unittest
import os
import numpy
from lsst.utils import getPackageDir
import lsst.utils.tests as utilsTests

from lsst.sims.utils import ObservationMetaData
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates, \
                                 pupilCoordinatesFromPixelCoordinates
from lsst.sims.GalSimInterface import GalSimDetector

class GalSimDetectorTest(unittest.TestCase):

    def setUp(self):
        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'),
                               'tests', 'cameraData')

        self.camera = ReturnCamera(baseDir)

        ra = 145.0
        dec = -73.0
        self.epoch = 2000.0
        mjd = 49250.0
        rotSkyPos = 45.0
        self.obs = ObservationMetaData(unrefractedRA=ra,
                                       unrefractedDec=dec,
                                       boundType='circle',
                                       boundLength=1.0,
                                       mjd=mjd,
                                       rotSkyPos=rotSkyPos)

        raPointing, \
        decPointing = observedFromICRS(numpy.array([numpy.radians(ra)]),
                                       numpy.array([numpy.radians(dec)]),
                                       obs_metadata=self.obs,
                                       epoch=self.epoch)

        self.ra = raPointing[0]
        self.dec = decPointing[0]


    def testContainsRaDec(self):
        """
        Test whether or not the method containsRaDec correctly identifies
        RA and Dec that fall inside and outside the detector
        """

        photParams = PhotometricParameters()
        gsdet = GalSimDetector(self.camera[0], self.camera, \
                               self.obs, self.epoch,
                               photParams=photParams)

        xxList = [gsdet.xMinPix, gsdet.xMaxPix]
        yyList = [gsdet.yMinPix, gsdet.yMaxPix]
        dxList = [-1.0, 1.0]
        dyList = [-1.0, 1.0]

        xPixList = []
        yPixList = []
        correctAnswer = []

        for xx, yy, dx, dy in zip(xxList, yyList, dxList, dyList):
            xPixList.append(xx)
            yPixList.append(yy)
            correctAnswer.append(True)

            xPixList.append(xx+dx)
            yPixList.append(yy)
            correctAnswer.append(False)

            xPixList.append(xx)
            yPixList.append(yy+dy)
            correctAnswer.append(False)


        nameList = [gsdet.name]*len(xPixList)
        xPixList = numpy.array(xPixList)
        yPixList = numpy.array(yPixList)

        raList, decList = raDecFromPixelCoordinates(xPixList, yPixList,
                                                    nameList,
                                                    camera=self.camera,
                                                    obs_metadata=self.obs,
                                                    epoch=self.epoch)

        testAnswer = gsdet.containsRaDec(raList, decList)

        for c, t in zip(correctAnswer, testAnswer):
            self.assertTrue(c is t)


    def testContainsPupilCoordinates(self):
        """
        Test whether or not the method containsRaDec correctly identifies
        RA and Dec that fall inside and outside the detector
        """

        photParams = PhotometricParameters()
        gsdet = GalSimDetector(self.camera[0], self.camera, \
                               self.obs, self.epoch,
                               photParams=photParams)

        xxList = [gsdet.xMinPix, gsdet.xMaxPix]
        yyList = [gsdet.yMinPix, gsdet.yMaxPix]
        dxList = [-1.0, 1.0]
        dyList = [-1.0, 1.0]

        xPixList = []
        yPixList = []
        correctAnswer = []

        for xx, yy, dx, dy in zip(xxList, yyList, dxList, dyList):
            xPixList.append(xx)
            yPixList.append(yy)
            correctAnswer.append(True)

            xPixList.append(xx+dx)
            yPixList.append(yy)
            correctAnswer.append(False)

            xPixList.append(xx)
            yPixList.append(yy+dy)
            correctAnswer.append(False)


        nameList = [gsdet.name]*len(xPixList)
        xPixList = numpy.array(xPixList)
        yPixList = numpy.array(yPixList)

        xPupilList, yPupilList = \
               pupilCoordinatesFromPixelCoordinates(xPixList, yPixList,
                                                    nameList,
                                                    camera=self.camera,
                                                    obs_metadata=self.obs,
                                                    epoch=self.epoch)


        testAnswer = gsdet.containsPupilCoordinates(xPupilList, yPupilList)

        for c, t in zip(correctAnswer, testAnswer):
            self.assertTrue(c is t)

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimDetectorTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
