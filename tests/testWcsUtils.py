import unittest
import os
import numpy
import lsst.utils.tests as utilsTests
import lsst.afw.geom as afwGeom
from lsst.utils import getPackageDir
from lsst.sims.utils import ObservationMetaData, haversine, arcsecFromRadians
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import _raDecFromPixelCoords
from lsst.sims.GalSimInterface.wcsUtils import tanWcsFromDetector, tanSipWcsFromDetector


class WcsTest(unittest.TestCase):

    def setUp(self):
        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        self.camera = ReturnCamera(baseDir)
        self.obs = ObservationMetaData(pointingRA=25.0, pointingDec=-10.0,
                                       boundType='circle', boundLength=1.0,
                                       mjd=49250.0, rotSkyPos=0.0)
        self.epoch = 2000.0


    def testTanWcs(self):
        """
        Test method to return a Tan WCS by generating a bunch of pixel coordinates
        in the undistorted TAN-PIXELS coordinate system.  Then, use sims_coordUtils
        to convert those pixel coordinates into RA and Dec.  Compare these to the
        RA and Dec returned by the WCS.  Demand agreement to witin 0.001 arcseconds.

        Note: if you use a bigger camera, it is possible to have disagreements of
        order a few milliarcseconds.
        """

        detector = self.camera[0]

        xPixList = []
        yPixList = []

        tanWcs = tanWcsFromDetector(detector, self.camera, self.obs, self.epoch)
        wcsRa = []
        wcsDec = []
        for xx in numpy.arange(0.0, 4001.0, 1000.0):
            for yy in numpy.arange(0.0, 4001.0, 1000.0):
                xPixList.append(xx)
                yPixList.append(yy)

                pt = afwGeom.Point2D(xx ,yy)
                skyPt = tanWcs.pixelToSky(pt).getPosition()
                wcsRa.append(skyPt.getX())
                wcsDec.append(skyPt.getY())

        wcsRa = numpy.radians(numpy.array(wcsRa))
        wcsDec = numpy.radians(numpy.array(wcsDec))

        xPixList = numpy.array(xPixList)
        yPixList = numpy.array(yPixList)

        raTest, decTest = _raDecFromPixelCoords(xPixList, yPixList,
                                                [detector.getName()]*len(xPixList),
                                                camera=self.camera, obs_metadata=self.obs,
                                                epoch=self.epoch)

        distanceList = arcsecFromRadians(haversine(raTest, decTest, wcsRa, wcsDec))
        maxDistance = distanceList.max()

        msg = 'maxError in tanWcs was %e ' % maxDistance
        self.assertTrue(maxDistance<0.001, msg=msg)


    def testTanSipWcs(self):
        """
        Test that tanSipWcsFromDetector works by fitting a TAN WCS and a TAN-SIP WCS to
        the a detector with distortions and verifying that the TAN-SIP WCS better approximates
        the truth.
        """

        detector = self.camera[0]
        tanWcs = tanWcsFromDetector(detector, self.camera, self.obs, self.epoch)
        tanSipWcs = tanSipWcsFromDetector(detector, self.camera, self.obs, self.epoch)

        tanWcsRa = []
        tanWcsDec = []
        tanSipWcsRa = []
        tanSipWcsDec = []

        xPixList = []
        yPixList = []
        for xx in numpy.arange(0.0, 4001.0, 1000.0):
            for yy in numpy.arange(0.0, 4001.0, 1000.0):
                xPixList.append(xx)
                yPixList.append(yy)

                pt = afwGeom.Point2D(xx ,yy)
                skyPt = tanWcs.pixelToSky(pt).getPosition()
                tanWcsRa.append(skyPt.getX())
                tanWcsDec.append(skyPt.getY())

                skyPt = tanSipWcs.pixelToSky(pt).getPosition()
                tanSipWcsRa.append(skyPt.getX())
                tanSipWcsDec.append(skyPt.getY())

        tanWcsRa = numpy.radians(numpy.array(tanWcsRa))
        tanWcsDec = numpy.radians(numpy.array(tanWcsDec))

        tanSipWcsRa = numpy.radians(numpy.array(tanSipWcsRa))
        tanSipWcsDec = numpy.radians(numpy.array(tanSipWcsDec))

        xPixList = numpy.array(xPixList)
        yPixList = numpy.array(yPixList)

        raTest, decTest = _raDecFromPixelCoords(xPixList, yPixList,
                                                [detector.getName()]*len(xPixList),
                                                camera=self.camera, obs_metadata=self.obs,
                                                epoch=self.epoch)

        tanDistanceList = arcsecFromRadians(haversine(raTest, decTest, tanWcsRa, tanWcsDec))
        tanSipDistanceList = arcsecFromRadians(haversine(raTest, decTest, tanSipWcsRa, tanSipWcsDec))

        maxDistanceTan = tanDistanceList.max()
        maxDistanceTanSip = tanSipDistanceList.max()

        msg = 'max error in TAN WCS %e; in TAN-SIP %e' % (maxDistanceTan, maxDistanceTanSip)
        self.assertTrue(maxDistanceTanSip<0.001, msg=msg)
        self.assertTrue(maxDistanceTan-maxDistanceTanSip>1.0e-10, msg=msg)


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(WcsTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
