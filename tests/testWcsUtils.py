import unittest
import os
import numpy
import lsst.utils.tests as utilsTests
import lsst.afw.geom as afwGeom
from lsst.utils import getPackageDir
from lsst.sims.utils import ObservationMetaData, haversine, arcsecFromRadians
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates
from lsst.sims.GalSimInterface.wcsUtils import nativeLonLatFromRaDec, raDecFromNativeLonLat, \
                                                _raDecFromNativeLonLat
from lsst.sims.GalSimInterface.wcsUtils import tanWcsFromDetector, tanSipWcsFromDetector

class NativeLonLatTest(unittest.TestCase):

    def testNativeLonLat(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations
        """

        raList = [0.0, 0.0, 0.0, 270.0]
        decList = [90.0, 90.0, 0.0, 0.0]

        raPointList = [0.0, 270.0, 270.0, 0.0]
        decPointList = [0.0, 0.0,0.0, 0.0]

        lonControlList = [180.0, 180.0, 90.0, 270.0]
        latControlList = [0.0, 0.0, 0.0, 0.0]

        for rr, dd, rp, dp, lonc, latc in \
        zip(raList, decList, raPointList, decPointList, lonControlList, latControlList):
            lon, lat = nativeLonLatFromRaDec(rr, dd, rp, dp)
            self.assertAlmostEqual(lon, lonc, 10)
            self.assertAlmostEqual(lat, latc, 10)


    def testNativeLonLatVector(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations (make sure it works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way)
        """

        raPoint = 145.0
        decPoint = -35.0

        nSamples = 100
        numpy.random.seed(42)
        raList = numpy.random.random_sample(nSamples)*360.0
        decList = numpy.random.random_sample(nSamples)*180.0 - 90.0

        lonList, latList = nativeLonLatFromRaDec(raList, decList, raPoint, decPoint)

        for rr, dd, lon, lat in zip(raList, decList, lonList, latList):
            lonControl, latControl = nativeLonLatFromRaDec(rr, dd, raPoint, decPoint)
            self.assertAlmostEqual(lat, latControl, 10)
            if numpy.abs(numpy.abs(lat) - 90.0)>1.0e-9:
                self.assertAlmostEqual(lon, lonControl, 10)


    def testRaDec(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec
        """
        numpy.random.seed(42)
        nSamples = 100
        raList = numpy.random.random_sample(nSamples)*360.0
        decList = numpy.random.random_sample(nSamples)*180.0 - 90.0
        raPointingList = numpy.random.random_sample(nSamples)*260.0
        decPointingList = numpy.random.random_sample(nSamples)*90.0 - 180.0

        for rr, dd, rp, dp in \
        zip(raList, decList, raPointingList, decPointingList):
            lon, lat = nativeLonLatFromRaDec(rr, dd, rp, dp)
            r1, d1 = raDecFromNativeLonLat(lon, lat, rp, dp)
            self.assertAlmostEqual(d1, dd, 10)
            if numpy.abs(numpy.abs(d1)-90.0)>1.0e-9:
               self.assertAlmostEqual(r1, rr, 10)

    def testRaDecVector(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec (make sure it works in a vectorized way)
        """
        numpy.random.seed(42)
        nSamples = 100
        latList = numpy.random.random_sample(nSamples)*360.0
        lonList = numpy.random.random_sample(nSamples)*180.0 - 90.0
        raPoint = 95.0
        decPoint = 75.0

        raList, decList = raDecFromNativeLonLat(lonList, latList, raPoint, decPoint)

        for lon, lat, ra0, dec0 in zip(lonList, latList, raList, decList):
            ra1, dec1 = raDecFromNativeLonLat(lon, lat, raPoint, decPoint)
            self.assertAlmostEqual(dec0, dec1, 10)
            if numpy.abs(numpy.abs(dec0)-90.0)>1.0e-9:
               self.assertAlmostEqual(ra0, ra1, 10)

class WcsTest(unittest.TestCase):

    def setUp(self):
        baseDir = os.path.join(getPackageDir('sims_GalSimInterface'), 'tests', 'cameraData')
        self.camera = ReturnCamera(baseDir)
        self.obs = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-10.0,
                                       boundType='circle', boundLength=1.0,
                                       mjd=49250.0, rotSkyPos=0.0)
        self.epoch = 2000.0

        self.raPointing, self.decPointing = observedFromICRS(numpy.array([self.obs._unrefractedRA]),
                                                             numpy.array([self.obs._unrefractedDec]),
                                                             obs_metadata=self.obs,
                                                             epoch=self.epoch)

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

        raTest, decTest = raDecFromPixelCoordinates(xPixList, yPixList,
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

        raTest, decTest = raDecFromPixelCoordinates(xPixList, yPixList,
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
    suites += unittest.makeSuite(NativeLonLatTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
