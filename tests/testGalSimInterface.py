from __future__ import with_statement
import os
import copy
import numpy
import unittest
import galsim
from collections import OrderedDict
import lsst.utils
import lsst.utils.tests as utilsTests
from lsst.sims.utils import arcsecFromRadians, radiansFromArcsec
from lsst.sims.photUtils import Bandpass, calcSkyCountsPerPixelForM5, LSSTdefaults, PhotometricParameters
from lsst.sims.coordUtils import pixelCoordsFromPupilCoords
from lsst.sims.catalogs.measures.instance import InstanceCatalog
from lsst.sims.catalogs.generation.utils import makePhoSimTestDB
from lsst.sims.utils import ObservationMetaData
from lsst.sims.GalSimInterface import GalSimGalaxies, GalSimStars, GalSimAgn, \
                                               SNRdocumentPSF, ExampleCCDNoise
from lsst.sims.catUtils.utils import calcADUwrapper, testGalaxyBulgeDBObj, testGalaxyDiskDBObj, \
                                     testGalaxyAgnDBObj, testStarsDBObj
import lsst.afw.image as afwImage

class testGalaxyCatalog(GalSimGalaxies):
    """
    Wraps the GalSimGalaxies class.  Adds columns to the output
    so that we can read the InstanceCatalog back in and verify that
    GalSim put the correct number of ADU in each FITS file.
    """
    bandpassNames = ['u', 'g', 'r']

    column_outputs = copy.deepcopy(GalSimGalaxies.column_outputs)
    column_outputs.remove('fitsFiles')
    column_outputs.append('magNorm')
    column_outputs.append('redshift')
    column_outputs.append('internalAv')
    column_outputs.append('internalRv')
    column_outputs.append('galacticAv')
    column_outputs.append('galacticRv')
    column_outputs.append('fitsFiles')

    PSF = SNRdocumentPSF()


class testStarCatalog(GalSimStars):
    """
    Wraps the GalSimStars class.  Adds columns to the output
    so that we can read the InstanceCatalog back in and verify that
    GalSim put the correct number of ADU in each FITS file.
    """
    bandpassNames = ['u', 'g', 'r']

    column_outputs = copy.deepcopy(GalSimStars.column_outputs)
    column_outputs.remove('fitsFiles')
    column_outputs.append('magNorm')
    column_outputs.append('redshift')
    column_outputs.append('internalAv')
    column_outputs.append('internalRv')
    column_outputs.append('galacticAv')
    column_outputs.append('galacticRv')
    column_outputs.append('fitsFiles')

    PSF = SNRdocumentPSF()

class testAgnCatalog(GalSimAgn):
    """
    Wraps the GalSimAgn class.  Adds columns to the output
    so that we can read the InstanceCatalog back in and verify that
    GalSim put the correct number of ADU in each FITS file.
    """
    bandpassNames = ['u', 'g', 'r']

    column_outputs = copy.deepcopy(GalSimAgn.column_outputs)
    column_outputs.remove('fitsFiles')
    column_outputs.append('magNorm')
    column_outputs.append('redshift')
    column_outputs.append('internalAv')
    column_outputs.append('internalRv')
    column_outputs.append('galacticAv')
    column_outputs.append('galacticRv')
    column_outputs.append('fitsFiles')

    PSF = SNRdocumentPSF()

class psfCatalog(testGalaxyCatalog):
    """
    Adds a PSF to testGalaxyCatalog
    """
    PSF = SNRdocumentPSF()

class backgroundCatalog(testGalaxyCatalog):
    """
    Add sky background but no noise to testGalaxyCatalog
    """
    PSF = SNRdocumentPSF()
    noise_and_background = ExampleCCDNoise(addNoise=False, seed=42)

class noisyCatalog(testGalaxyCatalog):
    """
    Adds a noise and sky background wrapper to testGalaxyCatalog
    """
    PSF = SNRdocumentPSF()
    noise_and_background = ExampleCCDNoise(seed=42)


class testFakeBandpassCatalog(testStarCatalog):
    """
    tests the GalSim interface on fake bandpasses
    """
    bandpassNames = ['x', 'y', 'z']

    bandpassDir = os.path.join(lsst.utils.getPackageDir('sims_catUtils'),'tests','testThroughputs')
    bandpassRoot = 'fakeFilter_'
    componentList = ['fakeM1.dat', 'fakeM2.dat']
    atmoTransmissionName = 'fakeAtmo.dat'
    skySEDname = 'fakeSky.dat'

class testFakeSedCatalog(testFakeBandpassCatalog):
    """
    tests the GalSim interface on fake seds and bandpasses
    """
    sedDir = os.path.join(lsst.utils.getPackageDir('sims_catUtils'),'tests','testSeds')

    def get_sedFilepath(self):
        """
        map the sedFilenames created by makePhoSimTestDB to the SEDs in
        in testSeds/
        """

        nameMap = {'km20_5750.fits_g40_5790':'fakeSed1.dat',
                   'm2.0Full.dat':'fakeSed2.dat',
                   'bergeron_6500_85.dat_6700':'fakeSed3.dat'}

        rawNames = self.column_by_name('sedFilename')
        return numpy.array([nameMap[nn] for nn in rawNames])


class GalSimInterfaceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dbName = 'galSimTestDB.db'
        if os.path.exists(cls.dbName):
            os.unlink(cls.dbName)

        displacedRA = numpy.array([72.0/3600.0])
        displacedDec = numpy.array([0.0])
        defaults = LSSTdefaults()
        cls.bandpassNameList = ['u', 'g', 'r', 'i', 'z', 'y']
        cls.m5 = defaults._m5.values()
        cls.seeing = defaults._seeing.values()
        cls.obs_metadata = makePhoSimTestDB(filename=cls.dbName, size=1,
                                            displacedRA=displacedRA,
                                            displacedDec=displacedDec,
                                            bandpass=cls.bandpassNameList,
                                            m5=cls.m5,
                                            seeing=cls.seeing)

        cls.driver = 'sqlite'



    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dbName):
            os.unlink(cls.dbName)

        del cls.dbName
        del cls.driver
        del cls.obs_metadata
        del cls.bandpassNameList
        del cls.m5
        del cls.seeing


    def getFilesAndBandpasses(self, catalog, nameRoot=None,
                                bandpassDir=os.path.join(lsst.utils.getPackageDir('throughputs'),'baseline'),
                                bandpassRoot='total_',):

        """
        Take a GalSimCatalog.  Return a list of fits files and and OrderedDict of bandpasses associated
        with that catalog

        @param [in] catalog is a GalSimCatalog instantiation

        @param [in] nameRoot is the nameRoot prepended to the fits files output by that catalog

        @param[in] bandpassDir is the directory where bandpass files can be found

        @param [in] bandpassRoot is the root of the name of the bandpass files

        @param [out] listOfFiles is a list of the names of the fits files written by this catalog

        @param [out] bandpassDict is an OrderedDict of Bandpass instantiations corresponding to the
        filters in this catalog.
        """

        #write the fits files
        catalog.write_images(nameRoot=nameRoot)

        #a list of bandpasses over which we are integraging
        listOfFilters = []
        listOfFiles = []

        #read in the names of all of the written fits files directly from the
        #InstanceCatalog's GalSimInterpreter
        #Use AFW to read in the FITS files and calculate the ADU
        for name in catalog.galSimInterpreter.detectorImages:
            if nameRoot is not None:
                name = nameRoot+'_'+name

            listOfFiles.append(name)

            if name[-6] not in listOfFilters:
                listOfFilters.append(name[-6])

        bandpassDict = OrderedDict()
        for filterName in listOfFilters:
            bandpassName=os.path.join(bandpassDir, bandpassRoot + filterName + '.dat')
            bandpass = Bandpass()
            bandpass.readThroughput(bandpassName)
            bandpassDict[filterName] = bandpass

        return listOfFiles, bandpassDict


    def catalogTester(self, catName=None, catalog=None, nameRoot=None,
                      bandpassDir=os.path.join(lsst.utils.getPackageDir('throughputs'),'baseline'),
                      bandpassRoot='total_',
                      sedDir=lsst.utils.getPackageDir('sims_sed_library')):
        """
        Reads in a GalSim Instance Catalog.  Writes the images from that catalog.
        Then reads those images back in.  Uses AFW to calculate the number of counts
        in each FITS image.  Reads in the InstanceCatalog associated with those images.
        Uses sims_photUtils code to calculate the ADU for each object on the FITS images.
        Verifies that the two independent calculations of counts agree (to within a tolerance,
        since the GalSim images are generated in a pseudo-random way).

        @param [in] catName is the name of the InstanceCatalog that has been written to disk

        @param [in] catalog is the actual InstanceCatalog instantiation

        @param [in] nameRoot is a string appended to the names of the FITS files being written

        @param [in] bandpassDir is the directory containing the bandpasses against which to test

        @param [in] bandpassRoot is the root of the name of the bandpass files, i.e.

            os.path.join(bandpassDir, bandpassRoot + bandpassName + '.dat')
        """

        #a dictionary of ADU for each FITS file as calculated by GalSim
        #(indexed on the name of the FITS file)
        galsimCounts = {}
        galsimPixels = {}

        #a dictionary of ADU for each FITS file as calculated by sims_photUtils
        #(indexed on the name of the FITS file)
        controlCounts = {}

        listOfFiles, bandpassDict = self.getFilesAndBandpasses(catalog, nameRoot=nameRoot,
                                                                 bandpassDir=bandpassDir,
                                                                 bandpassRoot=bandpassRoot)

        #read in the names of all of the written fits files directly from the
        #InstanceCatalog's GalSimInterpreter
        #Use AFW to read in the FITS files and calculate the ADU
        for name in listOfFiles:
            im = afwImage.ImageF(name)
            imArr = im.getArray()
            galsimCounts[name] = imArr.sum()
            galsimPixels[name] = imArr.shape[0]*imArr.shape[1]
            controlCounts[name] = 0.0
            os.unlink(name)

        if catalog.noise_and_background is not None and catalog.noise_and_background.addBackground:
            #calculate the expected skyCounts in each filter
            backgroundCounts = {}
            for filterName in bandpassDict.keys():
                cts = calcSkyCountsPerPixelForM5(catalog.obs_metadata.m5[filterName], bandpassDict[filterName],
                                             catalog.photParams, seeing=catalog.obs_metadata.seeing[filterName])

                backgroundCounts[filterName] = cts

            for name in controlCounts:
                filterName = name[-6]
                controlCounts[name] += backgroundCounts[filterName] * galsimPixels[name]

        #Read in the InstanceCatalog.  For each object in the catalog, use sims_photUtils
        #to calculate the ADU.  Keep track of how many ADU should be in each FITS file.
        with open(catName, 'r') as testFile:
            lines = testFile.readlines()
            for line in lines:
                if line[0] != '#':
                    gg = line.split(';')
                    sedName = gg[7]
                    magNorm = float(gg[13])
                    redshift = float(gg[14])
                    internalAv = float(gg[15])
                    internalRv = float(gg[16])
                    galacticAv = float(gg[17])
                    galacticRv = float(gg[18])
                    listOfFileNames = gg[19].split('//')
                    alreadyWritten = []

                    for name in listOfFileNames:

                        #guard against objects being written on one
                        #chip more than once
                        msg = '%s was written on %s more than once' % (sedName, name)
                        self.assertTrue(name not in alreadyWritten, msg=msg)
                        alreadyWritten.append(name)

                        #loop over all of the detectors on which an object fell
                        #(this is not a terribly great idea, since our conservative implementation
                        #of GalSimInterpreter._doesObjectImpingeOnDetector means that some detectors
                        #will be listed here even though the object does not illumine them)
                        for filterName in bandpassDict.keys():
                            chipName = name.replace(':','_')
                            chipName = chipName.replace(' ','_')
                            chipName = chipName.replace(',','_')
                            chipName = chipName.strip()

                            fullName = nameRoot+'_'+chipName+'_'+filterName+'.fits'

                            fullSedName = os.path.join(sedDir, sedName)

                            controlCounts[fullName] += calcADUwrapper(sedName=fullSedName, bandpass=bandpassDict[filterName],
                                                                        redshift=redshift, magNorm=magNorm,
                                                                        internalAv=internalAv, internalRv=internalRv,
                                                                        galacticAv=galacticAv, galacticRv=galacticRv)

            drawnDetectors = 0
            unDrawnDetectors = 0
            for ff in controlCounts:
                if controlCounts[ff] > 1000.0 and galsimCounts[ff] > 0.001:
                    #because, for really dim images, there could be enough
                    #statistical imprecision in the GalSim drawing routine
                    #to violate the condition below
                    drawnDetectors += 1
                    msg = 'controlCounts %e galsimCounts %e; %s ' % (controlCounts[ff], galsimCounts[ff],nameRoot)
                    if catalog.noise_and_background is not None and catalog.noise_and_background.addBackground:
                        msg += 'background per pixel %e pixels %e %s' % (backgroundCounts[ff[-6]], galsimPixels[ff],ff)

                    self.assertTrue(numpy.abs(controlCounts[ff] - galsimCounts[ff]) < 0.05*controlCounts[ff],
                                    msg=msg)
                elif galsimCounts[ff] > 0.001:
                    unDrawnDetectors += 1

            #to make sure we did not neglect more than one detector
            self.assertTrue(unDrawnDetectors<2)
            self.assertTrue(drawnDetectors>0)


    def compareCatalogs(self, cleanCatalog, noisyCatalog, gain, readnoise):
        """
        Read in two catalogs (one with noise, one without).  Compare the flux in each image
        pixel by pixel.  Make sure that the variation between the two is within expected limits.

        @param [in] cleanCatalog is the noiseless GalSimCatalog instantiation

        @param [in] noisyCatalog is the noisy GalSimCatalog instantiation

        @param [in] gain is the electrons per ADU for the GalSimCatalogs

        @param [in] readnoise is the electrons per pixel per exposure of the GalSimCatalogs]
        """

        cleanFileList, cleanBandpassDict = self.getFilesAndBandpasses(cleanCatalog, nameRoot='clean')
        noisyFileList, noisyBandpassDict = self.getFilesAndBandpasses(noisyCatalog, nameRoot='unclean')

        #calculate the expected skyCounts in each filter
        backgroundCounts = {}
        for filterName in noisyBandpassDict.keys():
            cts = calcSkyCountsPerPixelForM5(noisyCatalog.obs_metadata.m5[filterName], noisyBandpassDict[filterName],
                                         noisyCatalog.photParams, seeing=noisyCatalog.obs_metadata.seeing[filterName])

            backgroundCounts[filterName] = cts

        # Go through each image pixel by pixel.
        # Treat the value in the clean image as the mean intensity for that pixel.
        # Sum up (noisy-clean)^2/var
        # where var is determined by Poisson statistics from mean and readnoise.
        # Divide by the number of pixel
        # Make sure that this average does not deviate from unity

        countedImages = 0
        for noisyName, cleanName in zip(noisyFileList, cleanFileList):
            noisyIm = afwImage.ImageF(noisyName).getArray()
            cleanIm = afwImage.ImageF(cleanName).getArray()

            totalVar = 0.0
            totalMean = 0.0
            ct = 0.0

            self.assertEqual(cleanIm.shape[0], noisyIm.shape[0], msg='images not same shape')
            self.assertEqual(cleanIm.shape[1], noisyIm.shape[1], msg='images not same shape')

            var = cleanIm/gain + readnoise/(gain*gain)
            totalVar = (numpy.power(noisyIm-cleanIm,2)/var).sum()
            totalMean = cleanIm.sum()
            ct = float(cleanIm.shape[0]*cleanIm.shape[1])
            totalVar = totalVar/ct
            totalMean = totalMean/ct

            if totalMean>=100.0:
                countedImages += 1
                self.assertTrue(numpy.abs(totalVar-1.0) < 0.05)

            os.unlink(noisyName)
            os.unlink(cleanName)

        self.assertTrue(countedImages>0)


    def testGalaxyBulges(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of galaxy bulges
        """
        catName = 'testBulgeCat.sav'
        gals = testGalaxyBulgeDBObj(driver=self.driver, database=self.dbName)
        cat = testGalaxyCatalog(gals, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='bulge')
        if os.path.exists(catName):
            os.unlink(catName)


    def testGalaxyDisks(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of galaxy disks
        """
        catName = 'testDiskCat.sav'
        gals = testGalaxyDiskDBObj(driver=self.driver, database=self.dbName)
        cat = testGalaxyCatalog(gals, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='disk')
        if os.path.exists(catName):
            os.unlink(catName)


    def testStars(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of stars
        """
        catName = 'testStarCat.sav'
        stars = testStarsDBObj(driver=self.driver, database=self.dbName)
        cat = testStarCatalog(stars, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='stars')
        if os.path.exists(catName):
            os.unlink(catName)


    def testFakeBandpasses(self):
        """
        Test GalSim catalog with alternate bandpasses
        """
        catName = 'testFakeBandpassCat.sav'
        m5 = [22.0, 23.0, 25.0]
        seeing = [0.6, 0.5, 0.7]
        bandpassNames = ['x', 'y', 'z']
        obs_metadata = ObservationMetaData(
                       pointingRA=self.obs_metadata.pointingRA,
                       pointingDec=self.obs_metadata.pointingDec,
                       rotSkyPos=self.obs_metadata.rotSkyPos,
                       mjd=self.obs_metadata.mjd,
                       bandpassName=bandpassNames,
                       m5=m5,
                       seeing=seeing)

        stars = testStarsDBObj(driver=self.driver, database=self.dbName)
        cat = testFakeBandpassCatalog(stars, obs_metadata=obs_metadata)
        cat.write_catalog(catName)
        bandpassDir = os.path.join(lsst.utils.getPackageDir('sims_catUtils'),'tests','testThroughputs')
        self.catalogTester(catName=catName, catalog=cat, nameRoot='fakeBandpass',
                           bandpassDir=bandpassDir, bandpassRoot='fakeTotal_')

        if os.path.exists(catName):
            os.unlink(catName)

    def testFakeSeds(self):
        """
        Test GalSim catalog with alternate Seds
        """
        catName = 'testFakeSedCat.sav'
        m5 = [22.0, 23.0, 25.0]
        seeing = [0.6, 0.5, 0.7]
        bandpassNames = ['x', 'y', 'z']
        obs_metadata = ObservationMetaData(
                       pointingRA=self.obs_metadata.pointingRA,
                       pointingDec=self.obs_metadata.pointingDec,
                       rotSkyPos=self.obs_metadata.rotSkyPos,
                       mjd=self.obs_metadata.mjd,
                       bandpassName=bandpassNames,
                       m5=m5,
                       seeing=seeing)

        stars = testStarsDBObj(driver=self.driver, database=self.dbName)
        cat = testFakeSedCatalog(stars, obs_metadata=obs_metadata)
        cat.write_catalog(catName)
        bandpassDir = os.path.join(lsst.utils.getPackageDir('sims_catUtils'), 'tests', 'testThroughputs')
        sedDir = os.path.join(lsst.utils.getPackageDir('sims_catUtils'), 'tests', 'testSeds')
        self.catalogTester(catName=catName, catalog=cat, nameRoot='fakeBandpass',
                           bandpassDir=bandpassDir, bandpassRoot='fakeTotal_',
                           sedDir=sedDir)

        if os.path.exists(catName):
            os.unlink(catName)



    def testAgns(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of AGN
        """
        catName = 'testAgnCat.sav'
        agn = testGalaxyAgnDBObj(driver=self.driver, database=self.dbName)
        cat = testAgnCatalog(agn, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='agn')
        if os.path.exists(catName):
            os.unlink(catName)


    def testPSFimages(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of Galaxy bulges convolved
        with a PSF
        """
        catName = 'testPSFcat.sav'
        gals = testGalaxyBulgeDBObj(driver=self.driver, database=self.dbName)
        cat = psfCatalog(gals, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='psf')
        if os.path.exists(catName):
            os.unlink(catName)


    def testBackground(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of Galaxy bulges with
        a sky background
        """
        catName = 'testBackgroundCat.sav'
        gals = testGalaxyBulgeDBObj(driver=self.driver, database=self.dbName)
        cat = backgroundCatalog(gals, obs_metadata = self.obs_metadata)
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='background')
        if os.path.exists(catName):
            os.unlink(catName)


    def testNoisyCatalog(self):
        """
        Compare noisy and noiseless images drawn from the same catalog.
        Make sure that the pixel-by-pixel difference between the two is
        as expected from Poisson statistics.
        """
        noisyCatName = 'testNoisyCatalog.sav'
        cleanCatName = 'testCleanCatalog.sav'

        gals = testGalaxyBulgeDBObj(driver=self.driver, database=self.dbName)

        noisyCat = noisyCatalog(gals, obs_metadata=self.obs_metadata)
        cleanCat = backgroundCatalog(gals, obs_metadata=self.obs_metadata)

        noisyCat.write_catalog(noisyCatName)
        cleanCat.write_catalog(cleanCatName)

        self.compareCatalogs(cleanCat, noisyCat, PhotometricParameters().gain, PhotometricParameters().readnoise)

        if os.path.exists(noisyCatName):
            os.unlink(noisyCatName)
        if os.path.exists(cleanCatName):
            os.unlink(cleanCatName)


    def testNoise(self):
        """
        Test that ExampleCCDNoise puts the expected counts on an image
        by generating a flat image, adding noise and background to it,
        and calculating the variance of counts in the image.
        """

        lsstDefaults = LSSTdefaults()
        gain = 2.5
        readnoise = 6.0
        photParams=PhotometricParameters(gain=gain, readnoise=readnoise)
        img = galsim.Image(100,100)
        noise = ExampleCCDNoise(seed=42)
        m5 = 24.5
        bandpass = Bandpass()
        bandpass.readThroughput(os.path.join(lsst.utils.getPackageDir('throughputs'),'baseline','total_r.dat'))
        background = calcSkyCountsPerPixelForM5(m5, bandpass, seeing=lsstDefaults.seeing('r'),
                                            photParams=photParams)

        noisyImage = noise.addNoiseAndBackground(img, bandpass, m5=m5,
                                                 seeing=lsstDefaults.seeing('r'),
                                                 photParams=photParams)

        mean = 0.0
        var = 0.0
        for ix in range(1,101):
            for iy in range(1,101):
                mean += noisyImage(ix, iy)

        mean = mean/10000.0

        for ix in range(1,101):
            for iy in range(1,101):
                var += (noisyImage(ix, iy) - mean)*(noisyImage(ix, iy) - mean)

        var = var/9999.0

        varElectrons = background*gain + readnoise
        varADU = varElectrons/(gain*gain)

        msg = 'background %e mean %e ' % (background, mean)
        self.assertTrue(numpy.abs(background/mean - 1.0) < 0.05, msg=msg)

        msg = 'var %e varADU %e ; ratio %e ; background %e' % (var, varADU, var/varADU, background)
        self.assertTrue(numpy.abs(var/varADU - 1.0) < 0.05, msg=msg)


    def testMultipleImages(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images of multiple objects
        """
        dbName = 'galSimTestMultipleDB.db'
        driver = 'sqlite'

        if os.path.exists(dbName):
            os.unlink(dbName)

        displacedRA = numpy.array([72.0/3600.0, 55.0/3600.0, 75.0/3600.0])
        displacedDec = numpy.array([0.0, 15.0/3600.0, -15.0/3600.0])
        obs_metadata = makePhoSimTestDB(filename=dbName, size=1,
                                        displacedRA=displacedRA, displacedDec=displacedDec,
                                        bandpass=self.bandpassNameList,
                                        m5=self.m5, seeing=self.seeing)

        gals = testGalaxyBulgeDBObj(driver=driver, database=dbName)
        cat = testGalaxyCatalog(gals, obs_metadata=obs_metadata)
        catName = 'multipleCatalog.sav'
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='multiple')
        if os.path.exists(catName):
            os.unlink(catName)

        stars = testStarsDBObj(driver=driver, database=dbName)
        cat = testStarCatalog(stars, obs_metadata=obs_metadata)
        catName = 'multipleStarCatalog.sav'
        cat.write_catalog(catName)
        self.catalogTester(catName=catName, catalog=cat, nameRoot='multipleStars')
        if os.path.exists(catName):
            os.unlink(catName)

        if os.path.exists(dbName):
            os.unlink(dbName)


    def testCompoundFitsFiles(self):
        """
        Test that GalSimInterpreter puts the right number of counts on images containgin different types of objects
        """
        driver = 'sqlite'
        dbName1 = 'galSimTestCompound1DB.db'
        if os.path.exists(dbName1):
            os.unlink(dbName1)

        displacedRA = numpy.array([72.0/3600.0, 55.0/3600.0, 75.0/3600.0])
        displacedDec = numpy.array([0.0, 15.0/3600.0, -15.0/3600.0])
        obs_metadata1 = makePhoSimTestDB(filename=dbName1, size=1,
                                         displacedRA=displacedRA, displacedDec=displacedDec,
                                         bandpass=self.bandpassNameList,
                                         m5=self.m5, seeing=self.seeing)

        dbName2 = 'galSimTestCompound2DB.db'
        if os.path.exists(dbName2):
            os.unlink(dbName2)

        displacedRA = numpy.array([55.0/3600.0, 60.0/3600.0, 62.0/3600.0])
        displacedDec = numpy.array([-3.0/3600.0, 10.0/3600.0, 10.0/3600.0])
        obs_metadata2 = makePhoSimTestDB(filename=dbName2, size=1,
                                            displacedRA=displacedRA, displacedDec=displacedDec,
                                            bandpass=self.bandpassNameList,
                                            m5=self.m5, seeing=self.seeing)

        gals = testGalaxyBulgeDBObj(driver=driver, database=dbName1)
        cat1 = testGalaxyCatalog(gals, obs_metadata=obs_metadata1)
        catName = 'compoundCatalog.sav'
        cat1.write_catalog(catName)

        stars = testStarsDBObj(driver=driver, database=dbName2)
        cat2 = testStarCatalog(stars, obs_metadata=obs_metadata2)
        cat2.copyGalSimInterpreter(cat1)
        cat2.write_catalog(catName, write_header=False, write_mode='a')
        self.catalogTester(catName=catName, catalog=cat2, nameRoot='compound')

        if os.path.exists(dbName1):
            os.unlink(dbName1)
        if os.path.exists(dbName2):
            os.unlink(dbName2)
        if os.path.exists(catName):
            os.unlink(catName)


    def testPlacement(self):
        """
        Test that GalSimInterpreter puts objects on the right detectors.

        Do so by creating a catalog of 10 closely-packed stars.  Draw test FITS
        images of them using the GalSim Catalog infrastructure.  Draw control FITS
        images of the detectors in the camera, paranoidly including every star
        in every control image (GalSim contains code such that it will not
        actually add flux to an image in cases where we try to include a
        star that does not actually fall on a detector).  Compare that

        a) the fluxes of the test and control images agree within some tolerance

        b) the fluxes of control images that have no corresponding test image
        (i.e. detectors on which no star actually fell) are effectively zero
        """

        #generate the database
        numpy.random.seed(32)
        catSize = 10
        dbName = 'galSimPlacementTestDB.db'
        driver = 'sqlite'
        if os.path.exists(dbName):
            os.unlink(dbName)

        displacedRA = (-40.0 + numpy.random.sample(catSize)*(120.0))/3600.0
        displacedDec = (-20.0 + numpy.random.sample(catSize)*(80.0))/3600.0
        obs_metadata = makePhoSimTestDB(filename=dbName, displacedRA=displacedRA, displacedDec=displacedDec,
                                        bandpass=self.bandpassNameList,
                                        m5=self.m5, seeing=self.seeing)

        catName = 'testPlacementCat.sav'
        stars = testStarsDBObj(driver=driver, database=dbName)

        #create the catalog
        cat = testStarCatalog(stars, obs_metadata = obs_metadata)
        results = cat.iter_catalog()
        firstLine = True

        #iterate over the catalog, giving every star a chance to
        #illumine every detector
        controlImages = {}
        for i, line in enumerate(results):
            galSimType = line[0]
            xPupil = line[5]
            yPupil = line[6]


            majorAxis = line[8]
            minorAxis = line[9]
            sindex = line[10]
            halfLightRadius = line[11]
            positionAngle = line[12]
            if firstLine:
                sedList = cat._calculateGalSimSeds()
                for detector in cat.galSimInterpreter.detectors:
                    for bandpass in cat.galSimInterpreter.bandpasses:
                        controlImages['placementControl_' + \
                                      cat.galSimInterpreter._getFileName(detector=detector, bandpassName=bandpass)] = \
                                      cat.galSimInterpreter.blankImage(detector=detector)
                firstLine = False

            spectrum = galsim.SED(spec=lambda ll: numpy.interp(ll, sedList[i].wavelen, sedList[i].flambda), flux_type='flambda')
            for bp in cat.galSimInterpreter.bandpasses:
                bandpass = cat.galSimInterpreter.bandpasses[bp]
                for detector in cat.galSimInterpreter.detectors:
                    centeredObj = cat.galSimInterpreter.PSF.applyPSF(xPupil=xPupil, yPupil=yPupil, bandpass=bandpass)

                    xPix, yPix = pixelCoordsFromPupilCoords(numpy.array([radiansFromArcsec(xPupil)]),
                                                            numpy.array([radiansFromArcsec(yPupil)]),
                                                            chipNames = [detector.name],
                                                            camera = detector.afwCamera)

                    dx = xPix[0] - detector.xCenterPix
                    dy = yPix[0] - detector.yCenterPix
                    obj = centeredObj*spectrum
                    localImage = cat.galSimInterpreter.blankImage(detector=detector)
                    localImage = obj.drawImage(bandpass=bandpass, wcs=detector.wcs, method='phot',
                                               gain=detector.photParams.gain, image=localImage,
                                               offset=galsim.PositionD(dx, dy))

                    controlImages['placementControl_' + \
                                  cat.galSimInterpreter._getFileName(detector=detector, bandpassName=bp)] += \
                                  localImage

        for name in controlImages:
            controlImages[name].write(file_name=name)

        #write the test images using the catalog infrastructure
        testNames = cat.write_images(nameRoot='placementTest')

        #make sure that every test image has a corresponding control image
        for testName in testNames:
            controlName = testName.replace('Test', 'Control')
            msg = '%s has no counterpart ' % testName
            self.assertTrue(controlName in controlImages, msg=msg)

        #make sure that the test and control images agree to some tolerance
        ignored = 0
        zeroFlux = 0
        for controlName in controlImages:
            controlImage = afwImage.ImageF(controlName)
            controlFlux = controlImage.getArray().sum()

            testName = controlName.replace('Control', 'Test')
            if testName in testNames:
                testImage = afwImage.ImageF(testName)
                testFlux = testImage.getArray().sum()
                msg = '%s: controlFlux = %e, testFlux = %e' % (controlName, controlFlux, testFlux)
                if controlFlux>1000.0:
                    #the randomness of photon shooting means that faint images won't agree
                    self.assertTrue(numpy.abs(controlFlux/testFlux - 1.0)<0.1, msg=msg)
                else:
                    ignored += 1
            else:
                #make sure that controlImages that have no corresponding test image really do
                #have zero flux (because no star fell on them)
                zeroFlux += 1
                msg = '%s has flux %e but was not written by catalog' % (controlName, controlFlux)
                self.assertTrue(controlFlux<1.0, msg=msg)

        self.assertTrue(ignored<len(testNames)/2)
        self.assertTrue(zeroFlux>0)

        for testName in testNames:
            if os.path.exists(testName):
                os.unlink(testName)

        for controlName in controlImages:
            if os.path.exists(controlName):
                os.unlink(controlName)

        if os.path.exists(dbName):
            os.unlink(dbName)


    def testPSF(self):
        """
        This method will test that SNRdocumentPSF returns a PSF
        with the correct Full Width at Half Max
        """

        fwhm = 0.4 #in arc-seconds; make sure that it divides evenly by scale, so that rounding
                   #half integer numbers of pixels does not affect the unit test

        scale = 0.1 #arc-seconds per pixel

        psf = SNRdocumentPSF(fwhm=fwhm)
        image = psf._cached_psf.drawImage(scale=scale)
        xCenter = (image.getXMax() + image.getXMin())/2
        yCenter = (image.getYMax() + image.getYMin())/2

        maxValue = image(xCenter, yCenter) #because the default is to center GSObjects
        halfDex = int(numpy.round(0.5*fwhm/scale)) #the distance from the center corresponding to FWHM

        #Test that pixel combinations bracketing the expected FWHM value behave
        #the way we expect them to
        midP1 = image(xCenter+halfDex+1, yCenter)
        midM1 = image(xCenter+halfDex-1, yCenter)
        msg = '%e is not > %e ' % (midM1, 0.5*maxValue)
        self.assertTrue(midM1 > 0.5*maxValue, msg=msg)
        msg = '%e is not < %e ' % (midP1, 0.5*maxValue)
        self.assertTrue(midP1 < 0.5*maxValue, msg=msg)

        midValue = image(xCenter-halfDex, yCenter)
        midP1 = image(xCenter-halfDex-1, yCenter)
        midM1 = image(xCenter-halfDex+1, yCenter)
        msg = '%e is not > %e ' % (midM1, 0.5*maxValue)
        self.assertTrue(midM1 > 0.5*maxValue, msg=msg)
        msg = '%e is not < %e ' % (midP1, 0.5*maxValue)
        self.assertTrue(midP1 < 0.5*maxValue, msg=msg)

        midP1 = image(xCenter, yCenter+halfDex+1)
        midM1 = image(xCenter, yCenter+halfDex-1)
        msg = '%e is not > %e ' % (midM1, 0.5*maxValue)
        self.assertTrue(midM1 > 0.5*maxValue, msg=msg)
        msg = '%e is not < %e ' % (midP1, 0.5*maxValue)
        self.assertTrue(midP1 < 0.5*maxValue, msg=msg)

        midP1 = image(xCenter, yCenter-halfDex-1)
        midM1 = image(xCenter, yCenter-halfDex+1)
        msg = '%e is not > %e ' % (midM1, 0.5*maxValue)
        self.assertTrue(midM1 > 0.5*maxValue, msg=msg)
        msg = '%e is not < %e ' % (midP1, 0.5*maxValue)
        self.assertTrue(midP1 < 0.5*maxValue, msg=msg)


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(GalSimInterfaceTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
