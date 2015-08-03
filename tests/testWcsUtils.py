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

        raControlList = [180.0, 180.0, 90.0, 270.0]
        decControlList = [0.0, 0.0, 0.0, 0.0]

        for rr, dd, rp, dp, rc, dc in \
        zip(raList, decList, raPointList, decPointList, raControlList, decControlList):
            rt, dt = nativeLonLatFromRaDec(rr, dd, rp, dp)
            self.assertAlmostEqual(rt, rc, 10)
            self.assertAlmostEqual(dt, dc, 10)


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
