"""
This file defines the following classes:

GalSimInterpreter -- a class which takes objects passed by a GalSim Instance Catalog
(see galSimCatalogs.py) and uses GalSim to write them to FITS images.

GalSimDetector -- a class which stored information about a detector in a way that
GalSimInterpreter expects.
"""

import os
import numpy
import time
import galsim
from lsst.sims.utils import radiansFromArcsec
from lsst.sims.coordUtils import pixelCoordsFromPupilCoords

__all__ = ["GalSimInterpreter"]


class GalSimInterpreter(object):
    """
    This is the class which actually takes the objects contained in the GalSim Instance Catalog and converts them
    into FITS images.
    """

    def __init__(self, obs_metadata=None, detectors=None, bandpassDict=None, noiseWrapper=None, epoch=None, seed=None):

        """
        @param [in] obs_metadata is an instantiation of the ObservationMetaData class which
        carries data about this particular observation (telescope site and pointing information)

        @param [in] detectors is a list of GalSimDetectors for which we are drawing FITS images

        @param [in] bandpassNames is a list of the form ['u', 'g', 'r', 'i', 'z', 'y'] denoting
        the bandpasses for which we are drawing FITS images.

        @param [in] bandpassFiles is a list of paths to the bandpass data files corresponding to
        bandpassNames

        @param [in] noiseWrapper is an instantiation of a NoiseAndBackgroundBase
        class which tells the interpreter how to add sky noise to its images.

        @param [in] seed is an integer that will use to seed the random number generator
        used when drawing images (if None, GalSim will automatically create a random number
        generator seeded with the system clock)
        """

        self.obs_metadata = obs_metadata
        self.epoch = epoch
        self.PSF = None
        self.noiseWrapper = noiseWrapper

        if seed is not None:
            self._rng = galsim.UniformDeviate(seed)
        else:
            self._rng = None

        if detectors is None:
            raise RuntimeError("Will not create images; you passed no detectors to the GalSimInterpreter")

        self.detectors = detectors

        self.detectorImages = {} #this dict will contain the FITS images (as GalSim images)
        self.bandpasses = {} #this dict will contain the GalSim bandpass instantiations corresponding to the input bandpasses
        self.catSimBandpasses = None
        self.blankImageCache = {} #this dict will cache blank images associated with specific detectors.
                                  #It turns out that calling the image's constructor is more time-consuming than
                                  #returning a deep copy

        self.setBandpasses(bandpassDict=bandpassDict)


    def setBandpasses(self, bandpassDict):
        """
        Read in files containing bandpass data and store them in a dict of GalSim bandpass instantiations.

        @param [in] bandpassNames is a list of the names by which the bandpasses are to be referred
        i.e. ['u', 'g', 'r', 'i', 'z', 'y']

        @param [in] bandpassFiles is a list of paths to the files containing data for the bandpasses

        The bandpasses will be stored in the member variable self.bandpasses, which is a dict
        """

        self.catSimBandpasses = bandpassDict
        for bpname in bandpassDict:

            # 14 April 2015
            #For some reason, you need to pass in the bandpass as an instance of galsim.LookupTable.
            #If you pass a lambda function, image generation will get much slower and unit tests
            #will fail because too few counts are placed on images.
            bptest = galsim.Bandpass(throughput = galsim.LookupTable(x=bandpassDict[bpname].wavelen, f=bandpassDict[bpname].sb),
                                 wave_type='nm')

            self.bandpasses[bpname] = bptest

    def setPSF(self, PSF=None):
        """
        Set the PSF wrapper for this GalSimInterpreter

        @param [in] PSF is an instantiation of a class which inherits from PSFbase and defines _getPSF()
        """
        self.PSF=PSF

    def _getFileName(self, detector=None, bandpassName=None):
        """
        Given a detector and a bandpass name, return the name of the FITS file to be written

        @param [in] detector is an instantiation of GalSimDetector

        @param [in] bandpassName is a string i.e. 'u' denoting the filter being drawn

        The resulting filename will be detectorName_bandpassName.fits
        """
        return detector.fileName+'_'+bandpassName+'.fits'


    def _doesObjectImpingeOnDetector(self, xPupil=None, yPupil=None, detector=None,
                                     imgScale=None, nonZeroPixels=None):
        """
        Compare an astronomical object to a detector and determine whether or not that object will cast any
        light on that detector (in case the object is near the edge of a detector and will cast some
        incidental light onto it).

        This method is called by the method findAllDetectors.  findAllDetectors will generate a test image
        of an astronomical object.  It will find all of the pixels in that test image with flux above
        a certain threshold and pass that list of pixels into this method along with data characterizing
        the detector in question.  This method compares the pupil coordinates of those pixels with the pupil
        coordinate domain of the detector. If some of those pixels fall inside the detector, then this method
        returns True (signifying that the astronomical object does cast light on the detector).  If not, this
        method returns False.

        @param [in] xPupil the x pupil coordinate of the image's origin in arcseconds

        @param [in] yPupil the y pupil coordinate of the image's origin in arcseconds

        @param [in] detector an instantiation of GalSimDetector.  This is the detector against
        which we will compare the object.

        @param [in] nonZeroPixels is a numpy array of non-zero pixels from the test image referenced
        above.  nonZeroPixels[0] is their x coordinate (in pixel value).  nonZeroPixels[1] is
        ther y coordinate.

        @param [in] imgScale is the platescale of the test image in arcseconds per pixel
        """

        if detector is None:
            return False

        xPupilList = radiansFromArcsec(numpy.array([xPupil + ix*imgScale for ix in nonZeroPixels[0]]))
        yPupilList = radiansFromArcsec(numpy.array([yPupil + iy*imgScale for iy in nonZeroPixels[1]]))

        answer = detector.containsPupilCoordinates(xPupilList, yPupilList)

        if True in answer:
            return True
        else:
            return False


    def findAllDetectors(self, gsObject):

        """
        Find all of the detectors on which a given astronomical object casts light.

        This method works by drawing a test image of the astronomical object and comparing
        the pixels in that image with flux above a certain threshold value to the pixel
        domains of the detectors in the camera.  Any detectors which overlap these
        'active' pixels are considered illumined by the object.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [out] outputString is a string indicating which chips the object illumines
        (suitable for the GalSim InstanceCatalog classes)

        @param [out] outputList is a list of detector instantiations indicating which
        detectors the object illumines

        @param [out] centeredObjDict is a dict of GalSim Objects centered on the chip, one for
        each bandpass (in case the object is convolved with a wavelength dependent PSF).
        The dict will be keyed to the bandpassName values stored in self.bandpasses
        (i.e. 'u', 'g', 'r', 'i', 'z', 'y' for default LSST behavior)

        Note: parameters that only apply to Sersic profiles will be ignored in the case of
        pointSources, etc.
        """

        outputString = ''
        outputList = []
        centeredObjDict = {}
        centeredObj = None
        testScale = 0.1

        for bandpassName in self.bandpasses:
            if centeredObj is None or (self.PSF is not None and self.PSF.wavelength_dependent):
                #create a GalSim Object centered on the chip.  Re-create it for each bandpass if
                #it is convolved with a wavelength-dependent PSF

                centeredObj = self.createCenteredObject(gsObject, bandpassName=bandpassName)

            #for output; to be used by self.drawObject()
            centeredObjDict[bandpassName] = centeredObj

        nTests = 0
        for bandpassName in self.bandpasses:
            goOn = False
            if nTests==0 or (self.PSF is not None and self.PSF.wavelength_dependent):

                #if we have already decided that the object illumines all detectors,
                #there is no point in going on
                for dd in self.detectors:
                    if dd not in outputList:
                        goOn = True
                        break

            nTests += 1
            if goOn:
                centeredObj = centeredObjDict[bandpassName]

                if centeredObj is None:
                    return

                #4 March 2015
                #create a test image of the object to compare against the pixel
                #domains of each detector.  Use photon shooting rather than real space integration
                #for reasons of speed.  A flux of 1000 photons ought to be enough to plot the true
                #extent of the object, but this is just a guess.
                centeredImage = centeredObj.drawImage(scale=testScale, method='phot', n_photons=1000, rng=self._rng)
                xmax = testScale * (centeredImage.getXMax()/2) + gsObject.xPupilArcsec
                xmin = testScale * (-1*centeredImage.getXMax()/2) + gsObject.xPupilArcsec
                ymax = testScale * (centeredImage.getYMax()/2) + gsObject.yPupilArcsec
                ymin = testScale *(-1*centeredImage.getYMin()/2) + gsObject.yPupilArcsec

                #first assemble a list of detectors which have any hope
                #of overlapping the test image
                viableDetectors = []
                for dd in self.detectors:
                    xOverLaps = False
                    if xmax > dd.xMinArcsec and xmax < dd.xMaxArcsec:
                        xOverLaps = True
                    elif xmin > dd.xMinArcsec and xmin < dd.xMaxArcsec:
                        xOverLaps = True
                    elif xmin < dd.xMinArcsec and xmax > dd.xMaxArcsec:
                        xOverLaps = True

                    yOverLaps = False
                    if ymax > dd.yMinArcsec and ymax < dd.yMaxArcsec:
                        yOverLaps = True
                    elif ymin > dd.yMinArcsec and ymin < dd.yMaxArcsec:
                        yOverLaps = True
                    elif ymin < dd.yMinArcsec and ymax > dd.yMaxArcsec:
                        yOverLaps = True

                    if xOverLaps and yOverLaps and dd not in outputList:
                        viableDetectors.append(dd)


                if len(viableDetectors)>0:

                    #Find the pixels that have a flux greater than 0.001 times the flux of
                    #the central pixel (remember that the object is centered on the test image)
                    maxPixel = centeredImage(centeredImage.getXMax()/2, centeredImage.getYMax()/2)
                    activePixels = numpy.where(centeredImage.array>maxPixel*0.001)

                    #Find the bounds of those active pixels in pixel coordinates
                    xmin = testScale * (activePixels[0].min() - centeredImage.getXMax()/2) + gsObject.xPupilArcsec
                    xmax = testScale * (activePixels[0].max() - centeredImage.getXMax()/2) + gsObject.xPupilArcsec
                    ymin = testScale * (activePixels[1].min() - centeredImage.getYMax()/2) + gsObject.yPupilArcsec
                    ymax = testScale * (activePixels[1].max() - centeredImage.getYMax()/2) + gsObject.yPupilArcsec

                    #find all of the detectors that overlap with the bounds of the active pixels.
                    for dd in viableDetectors:
                        xOverLaps = False
                        if xmax > dd.xMinArcsec and xmax < dd.xMaxArcsec:
                            xOverLaps = True
                        elif xmin > dd.xMinArcsec and xmin < dd.xMaxArcsec:
                            xOverLaps = True
                        elif xmin < dd.xMinArcsec and xmax > dd.xMaxArcsec:
                            xOverLaps = True

                        yOverLaps = False
                        if ymax > dd.yMinArcsec and ymax < dd.yMaxArcsec:
                            yOverLaps = True
                        elif ymin > dd.yMinArcsec and ymin < dd.yMaxArcsec:
                            yOverLaps = True
                        elif ymin < dd.yMinArcsec and ymax > dd.yMaxArcsec:
                            yOverLaps = True

                        #specifically test that these overlapping detectors do contain active pixels
                        if xOverLaps and yOverLaps:
                            if self._doesObjectImpingeOnDetector(xPupil=gsObject.xPupilArcsec - centeredImage.getXMax()*testScale/2.0,
                                                                 yPupil=gsObject.yPupilArcsec - centeredImage.getYMax()*testScale/2.0,
                                                                 detector=dd, imgScale=centeredImage.scale,
                                                                 nonZeroPixels=activePixels):

                                if outputString != '':
                                    outputString += '//'
                                outputString += dd.name
                                outputList.append(dd)

        if outputString == '':
            outputString = None

        return outputString, outputList, centeredObjDict


    def blankImage(self, detector=None):
        """
        Draw a blank image associated with a specific detector.  The image will have the correct size
        for the given detector.

        param [in] detector is an instantiation of GalSimDetector
        """

        #in order to speed up the code (by a factor of ~2), this method
        #only draws a new blank image the first time it is called on a
        #given detector.  It then caches the blank images it has drawn and
        #uses GalSim's copy() method to return copies of cached blank images
        #whenever they are called for again.

        if detector.name in self.blankImageCache:
            return self.blankImageCache[detector.name].copy()
        else:
            image = galsim.Image(detector.xMaxPix-detector.xMinPix+1, detector.yMaxPix-detector.yMinPix+1, \
                                 wcs=detector.wcs)

            self.blankImageCache[detector.name] = image
            return image.copy()

    def drawObject(self, gsObject):
        """
        Draw an astronomical object on all of the relevant FITS files.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject
        class carrying all of the information for the object whose image
        is to be drawn

        @param [out] outputString is a string denoting which detectors the astronomical
        object illumines, suitable for output in the GalSim InstanceCatalog
        """

        t0 = time.clock()
        #find the detectors which the astronomical object illumines
        outputString, \
        detectorList, \
        centeredObjDict = self.findAllDetectors(gsObject)
        t_det = time.clock()

        if gsObject.sed is None or len(detectorList) == 0:
            #there is nothing to draw
            return outputString

        #go through the list of detector/bandpass combinations and initialize
        #all of the FITS files we will need (if they have not already been initialized)
        for detector in detectorList:
            for bandpassName in self.bandpasses:
                name = self._getFileName(detector=detector, bandpassName=bandpassName)
                if name not in self.detectorImages:
                    self.detectorImages[name] = self.blankImage(detector=detector)
                    if self.noiseWrapper is not None:
                        #Add sky background and noise to the image
                        self.detectorImages[name] = self.noiseWrapper.addNoiseAndBackground(self.detectorImages[name],
                                                                              bandpass=self.catSimBandpasses[bandpassName],
                                                                              m5=self.obs_metadata.m5[bandpassName],
                                                                              FWHMeff=self.obs_metadata.seeing[bandpassName],
                                                                              photParams=detector.photParams)

        spectrum = galsim.SED(spec = lambda ll: numpy.interp(ll, gsObject.sed.wavelen, gsObject.sed.flambda),
                              flux_type='flambda')

        for bandpassName in self.bandpasses:

            #create a new object if one has not already been created or if the PSF is wavelength
            #dependent (in which case, each filter is going to need its own initialized object)
            centeredObj = centeredObjDict[bandpassName]
            if centeredObj is None:
                return outputString

            for detector in detectorList:

                name = self._getFileName(detector=detector, bandpassName=bandpassName)

                xPix, yPix = pixelCoordsFromPupilCoords(numpy.array([gsObject.xPupilRadians]),
                                                        numpy.array([gsObject.yPupilRadians]),
                                                        chipNames=[detector.name],
                                                        camera=detector.afwCamera)

                obj = centeredObj.copy()

                #convolve the object's shape profile with the spectrum
                t_before_draw = time.clock()
                obj = obj*spectrum
                localImage = self.blankImage(detector=detector)
                time_before_draw_image = time.clock()
                localImage = obj.drawImage(bandpass=self.bandpasses[bandpassName], wcs=detector.wcs,
                                           method='phot', gain=detector.photParams.gain, image=localImage,
                                           offset=galsim.PositionD(xPix[0]-detector.xCenterPix, yPix[0]-detector.yCenterPix),
                                           rng=self._rng)
                time_drawImage = time.clock()-time_before_draw_image

                self.detectorImages[name] += localImage
                time_total_drawing = time.clock()-t_before_draw

        t_done = time.clock()
        t_total = t_done-t0
        print 'total drawing ',t_total,' time finding ',(t_det-t0)/t_total, \
        ' actual ',(time_total_drawing)/t_total,' just drawImage ',time_drawImage/t_total

        return outputString

    def drawPointSource(self, gsObject, bandpass=None):
        """
        Draw an image of a point source.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] bandpass is an instantiation of the galsim.Bandpass class characterizing
        the bandpass over which we are integrating (in case the PSF is wavelength dependent)
        """

        if self.PSF is None:
            raise RuntimeError("Cannot draw a point source in GalSim without a PSF")

        return self.PSF.applyPSF(xPupil=gsObject.xPupilArcsec, yPupil=gsObject.yPupilArcsec, bandpass=bandpass)

    def drawSersic(self, gsObject, bandpass=None):
        """
        Draw the image of a Sersic profile.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] bandpass is an instantiation of the galsim.Bandpass class characterizing
        the bandpass over which we are integrating (in case the PSF is wavelength dependent)
        """

        #create a Sersic profile
        centeredObj = galsim.Sersic(n=float(gsObject.sindex), half_light_radius=float(gsObject.halfLightRadiusArcsec))

        # Turn the Sersic profile into an ellipse
        # Subtract pi/2 from the position angle, because GalSim sets position angle=0
        # aligned with East, rather than North
        centeredObj = centeredObj.shear(q=gsObject.minorAxisRadians/gsObject.majorAxisRadians, \
                                        beta=(0.5*numpy.pi-gsObject.positionAngleRadians)*galsim.radians)
        if self.PSF is not None:
            centeredObj = self.PSF.applyPSF(xPupil=gsObject.xPupilArcsec, yPupil=gsObject.yPupilArcsec, obj=centeredObj,
                                            bandpass=bandpass)

        return centeredObj

    def createCenteredObject(self, gsObject, bandpassName=None):
        """
        Create a centered GalSim Object (i.e. if we were just to draw this object as an image,
        the object would be centered on the frame)

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] bandpassName is the tag indicating the bandpass (i.e. 'u', 'g', 'r', 'i', 'z', or 'y')

        @param [out] a GalSim Object suitable to base an image off of (but centered on the frame)

        Note: parameters that obviously only apply to Sersic profiles will be ignored in the case
        of point sources
        """

        if gsObject.galSimType == 'sersic':
            centeredObj = self.drawSersic(gsObject,
                                          bandpass=self.bandpasses[bandpassName])

        elif gsObject.galSimType == 'pointSource':
            centeredObj = self.drawPointSource(gsObject, bandpass=self.bandpasses[bandpassName])
        else:
            print "Apologies: the GalSimInterpreter does not yet have a method to draw "
            print gsObject.galSimType
            print " objects\n"
            centeredObj = None

        return centeredObj


    def writeImages(self, nameRoot=None):
        """
        Write the FITS files to disk.

        @param [in] nameRoot is a string that will be prepended to the names of the output
        FITS files.  The files will be named like

        @param [out] namesWritten is a list of the names of the FITS files written

        nameRoot_detectorName_bandpassName.fits

        myImages_R_0_0_S_1_1_y.fits is an example of an image for an LSST-like camera with
        nameRoot = 'myImages'
        """
        namesWritten = []
        for name in self.detectorImages:
            if nameRoot is not None:
                fileName = nameRoot+'_'+name
            else:
                fileName = name
            self.detectorImages[name].write(file_name=fileName)
            namesWritten.append(fileName)

        return namesWritten

