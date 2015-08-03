import unittest
import eups
import os
import numpy
import lsst.utils.tests as utilsTests
from lsst.sims.utils import ObservationMetaData, haversine
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.GalSimInterface.wcsUtils import nativeLonLatFromRaDec, raDecFromNativeLonLat

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
        baseDir = os.path.join(eups.productDir('sims_GalSimInterface'), 'tests', 'cameraData')
        self.camera = ReturnCamera(baseDir)
        self.obs = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-10.0,
                                       boundType='circle', boundLength=1.0,
                                       mjd=49250.0, rotSkyPos=0.0)

        self.epoch = 2000.0


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
