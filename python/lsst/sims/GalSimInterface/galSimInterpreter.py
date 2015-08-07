"""
This file defines the following classes:

GalSimInterpreter -- a class which takes objects passed by a GalSim Instance Catalog
(see galSimCatalogs.py) and uses GalSim to write them to FITS images.

GalSimDetector -- a class which stored information about a detector in a way that
GalSimInterpreter expects.
"""

import os
import numpy
import galsim
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.sims.utils import arcsecFromRadians, radiansFromArcsec
from lsst.sims.coordUtils import observedFromICRS, raDecFromPixelCoordinates, \
                                 calculatePixelCoordinates
from lsst.sims.photUtils import LSSTdefaults
from lsst.sims.GalSimInterface.wcsUtils import tanSipWcsFromDetector

__all__ = ["GalSimInterpreter", "GalSimDetector"]


class GalSim_afw_TanSipWCS(galsim.wcs.CelestialWCS):

    def __init__(self, afwDetector, afwCamera, obs_metadata, epoch):

        tanSipWcs = tanSipWcsFromDetector(afwDetector, afwCamera, obs_metadata, epoch)

        self.afwDetector = afwDetector
        self.afwCamera = afwCamera
        self.obs_metadata = obs_metadata
        self.epoch = epoch

        self.fitsHeader = tanSipWcs.getFitsMetadata()
        self.keywordList = self.fitsHeader.getOrderedNames()

        self.crpix1 = self.fitsHeader.get("CRPIX1")
        self.crpix2 = self.fitsHeader.get("CRPIX2")

        self.afw_crpix1 = self.crpix1
        self.afw_crpix2 = self.crpix2

        self.crval1 = self.fitsHeader.get("CRVAL1")
        self.crval2 = self.fitsHeader.get("CRVAL2")

        self.origin = galsim.PositionD(x=self.crpix1, y=self.crpix2)


    def _radec(self, x, y):
        """
        Convert pixel coordinates into ra, dec coordinates.
        x and y already have crpix1 and crpix2 subtracted from them.
        Return ra, dec in radians.
        """

        chipNameList = [self.afwDetector.getName()]

        if type(x) is numpy.ndarray:
            chipNameList = chipNameList * len(x)

        ra, dec = raDecFromPixelCoordinates(x + self.afw_crpix1, y + self.afw_crpix2, chipNameList,
                                            camera=self.afwCamera,
                                            obs_metadata=self.obs_metadata,
                                            epoch=self.epoch)
        if type(x) is numpy.ndarray:
            return (ra, dec)
        else:
            return (ra[0], dec[0])


    def _xy(self, ra, dec):
        """
        convert ra, dec in radians into x, y with crpix subtracted
        """

        chipNameList = [self.afwDetector.getName()]

        if type(ra) is numpy.ndarray:
            chipNameLIst = chipNameList * len(ra)

        xx, yy = calculatePixelCoordiantes(ra=ra, dec=dec, chipNames=chipNameList,
                                            obs_metadata=self.obs_metadata,
                                            epoch=self.epoch,
                                            camera = self.afwCamera)

        if type(ra) is numpy.ndarray:
            return (xx-self.crpix1, yy-self.crpix2)
        else:
            return (xx[0]-self.crpix1, yy-self.crpix2)


    def _newOrigin(self, origin):
        _newWcs = GalSim_afw_TanSipWCS(self.afwDetector, self.afwCamera, self.obs_metadata, self.epoch)
        _newWcs.crpix1 = origin.x
        _newWcs.crpix2 = origin.y
        _newWcs.fitsHeader.set('CRPIX1', origin.x)
        _newWcs.fitsHeader.set('CRPIX2', origin.y)
        return _newWcs


    def _writeHeader(self, header, bounds):
        for key in self.keywordList:
            header[key] = self.fitsHeader.get(key)



class GalSimDetector(object):
    """
    This class stores information about individual detectors for use by the GalSimInterpreter
    """

    def __init__(self, afwDetector, afwCamera, obs_metadata, epoch, photParams=None):
        """
        @param [in] afwDetector is an instaniation of afw.cameraGeom.Detector

        @param [in] afwCamera is an instantiation of afw.cameraGeom.camera

        @param [in] photParams is an instantiation of the PhotometricParameters class that carries
        details about the photometric response of the telescope.

        This class will generate its own internal variable self.fileName which is
        the name of the detector as it will appear in the output FITS files
        """

        if photParams is None:
            raise RuntimeError("You need to specify an instantiation of PhotometricParameters " +
                               "when constructing a GalSimDetector")

        self.name = afwDetector.getName()
        self.afwDetector = afwDetector
        self.afwCamera = afwCamera
        self.obs_metadata = obs_metadata
        self.epoch = epoch

        bbox = afwDetector.getBBox()
        self.xMinPix = bbox.getMinX()
        self.xMaxPix = bbox.getMaxX()
        self.yMinPix = bbox.getMinY()
        self.yMaxPix = bbox.getMaxY()

        self.bbox = afwGeom.Box2D(bbox)

        self.wcs = GalSim_afw_TanSipWCS(self.afwDetector, self.afwCamera, self.obs_metadata, self.epoch)

        self.pupilSystem = afwDetector.makeCameraSys(PUPIL)
        self.pixelSystem = afwDetector.makeCameraSys(PIXELS)

        centerPoint = afwDetector.getCenter(FOCAL_PLANE)
        centerPupil = afwCamera.transform(centerPoint, self.pupilSystem).getPoint()
        self.xCenter = arcsecFromRadians(centerPupil.getX())
        self.yCenter = arcsecFromRadians(centerPupil.getY())
        centerPixel = afwCamera.transform(centerPoint, self.pixelSystem).getPoint()
        self.xCenterPix = centerPixel.getX()
        self.yCenterPix = centerPixel.getY()

        ra, dec = raDecFromPixelCoordinates([self.xCenterPix], [self.yCenterPix], [self.name],
                                            camera=afwCamera, obs_metadata=obs_metadata,
                                            epoch=epoch)

        self.raCenter = ra[0]
        self.decCenter = dec[0]

        self.xMin = None
        self.yMin = None
        self.xMax = None
        self.yMax = None
        cornerPointList = afwDetector.getCorners(FOCAL_PLANE)
        for cornerPoint in cornerPointList:
            cameraPoint = afwDetector.makeCameraPoint(cornerPoint, FOCAL_PLANE)
            cameraPointPupil = afwCamera.transform(cameraPoint, self.pupilSystem).getPoint()

            xx = arcsecFromRadians(cameraPointPupil.getX())
            yy = arcsecFromRadians(cameraPointPupil.getY())
            if self.xMin is None or xx<self.xMin:
                self.xMin = xx
            if self.xMax is None or xx>self.xMax:
                self.xMax = xx
            if self.yMin is None or yy<self.yMin:
                self.yMin = yy
            if self.yMax is None or yy>self.yMax:
                self.yMax = yy


        self.photParams = photParams
        self.fileName = self._getFileName()


    def _getFileName(self):
        """
        Format the name of the detector to add to the name of the FITS file
        """
        detectorName = self.name
        detectorName = detectorName.replace(',','_')
        detectorName = detectorName.replace(':','_')
        detectorName = detectorName.replace(' ','_')

        name = detectorName
        return name


    def pixelCoordinatesFromRaDec(self, ra, dec):
        """
        Convert RA, Dec into pixel coordinates on this detector

        @param [in] ra is a numpy array or a float indicating RA in radians

        @param [in] dec is a numpy array or a float indicating Dec in radians

        @param [out] xPix is a numpy array indicating the x pixel coordinate

        @param [out] yPix is a numpy array indicating the y pixel coordinate
        """

        nameList = [self.name]
        if type(ra) is numpy.ndarray:
            nameList = nameList*len(ra)
            raLocal=ra
            decLocal=dec
        else:
            raLocal = numpy.array([ra])
            decLocal = numpy.array([dec])

        xPix, yPix = calculatePixelCoordinates(ra=raLocal, dec=decLocal, chipNames=nameList,
                                               obs_metadata=self.obs_metadata,
                                               epoch=self.epoch,
                                               camera=self.afwCamera)

        return xPix, yPix



    def pixelCoordinatesFromPupilCoordinates(self, xPupil, yPupil):
        """
        Convert pupil coordinates into pixel coordinates on this detector

        @param [in] xPupil is a numpy array or a float indicating x pupil coordinates
        in radians

        @param [in] yPupil a numpy array or a float indicating y pupil coordinates
        in radians

        @param [out] xPix is a numpy array indicating the x pixel coordinate

        @param [out] yPix is a numpy array indicating the y pixel coordinate
        """

        nameList = [self.name]
        if type(xPupil) is numpy.ndarray:
            nameList = nameList*len(xPupil)
            xp = xPupil
            yp = yPupil
        else:
            xp = numpy.array([xPupil])
            yp = numpy.array([yPupil])

        xPix, yPix = calculatePixelCoordinates(xPupil=xp, yPupil=yp, chipNames=nameList,
                                               obs_metadata=self.obs_metadata,
                                               epoch=self.epoch,
                                               camera=self.afwCamera)

        return xPix, yPix


    def containsRaDec(self, ra, dec):
        """
        Does a given RA, Dec fall on this detector?

        @param [in] ra is a numpy array or a float indicating RA in radians

        @param [in] dec is a numpy array or a float indicating Dec in radians

        @param [out] answer is an array of booleans indicating whether or not
        the corresponding RA, Dec pair falls on this detector
        """

        xPix, yPix = self.pixelCoordinatesFromRaDec(ra, dec)
        points = [afwGeom.Point2D(xx, yy) for xx, yy in zip(xPix, yPix)]
        answer = [self.bbox.contains(pp) for pp in points]
        return answer


    def containsPupilCoordinates(self, xPupil, yPupil):
        """
        Does a given set of pupil coordinates fall on this detector?

        @param [in] xPupil is a numpy array or a float indicating x pupil coordinates
        in radians

        @param [in] yPupuil is a numpy array or a float indicating y pupil coordinates
        in radians

        @param [out] answer is an array of booleans indicating whether or not
        the corresponding RA, Dec pair falls on this detector
        """
        xPix, yPix = self.pixelCoordinatesFromPupilCoordinates(xPupil, yPupil)
        points = [afwGeom.Point2D(xx, yy) for xx, yy in zip(xPix, yPix)]
        answer = [self.bbox.contains(pp) for pp in points]
        return answer


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
        self._LSSTdefaults = LSSTdefaults()
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


    def findAllDetectors(self, galSimType=None, xPupil=None,
                   yPupil=None, ra=None, dec=None, halfLightRadius=None, minorAxis=None, majorAxis=None,
                   positionAngle=None, sindex=None):

        """
        Find all of the detectors on which a given astronomical object casts light.

        This method works by drawing a test image of the astronomical object and comparing
        the pixels in that image with flux above a certain threshold value to the pixel
        domains of the detectors in the camera.  Any detectors which overlap these
        'active' pixels are considered illumined by the object.

        @param [in] galSimType is a string denoting the type of object (i.e. 'sersic' or 'pointSource')

        @param [in] xPupil is the x pupil coordinate of the object in radians

        @param [in] yPupil is the y pupil coordinate of the object in radians

        @param [in] ra is the RA coordinate of the object in radians

        @param [in] dec is the Dec coordinate of the object in radians

        @param [in] halfLightRadius is the half light radius of the object in radians

        @param [in] minorAxis is the semi-minor axis of the object in radians

        @param [in] majorAxis is the semi-major axis of the object in radians

        @param [in] positionAngle is the position angle of the object in radians

        @param [in] sindex is the Sersic index of the object's profile

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
        xp = arcsecFromRadians(xPupil)
        yp = arcsecFromRadians(yPupil)
        centeredObj = None
        testScale = 0.1

        for bandpassName in self.bandpasses:
            if centeredObj is None or (self.PSF is not None and self.PSF.wavelength_dependent):
                #create a GalSim Object centered on the chip.  Re-create it for each bandpass if
                #it is convolved with a wavelength-dependent PSF

                centeredObj = self.createCenteredObject(galSimType=galSimType,
                                                        xPupil=xPupil, yPupil=yPupil,
                                                        bandpassName=bandpassName,
                                                        sindex=sindex, halfLightRadius=halfLightRadius,
                                                        positionAngle=positionAngle,
                                                        minorAxis=minorAxis, majorAxis=majorAxis)

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
                xmax = testScale * (centeredImage.getXMax()/2) + xp
                xmin = testScale * (-1*centeredImage.getXMax()/2) + xp
                ymax = testScale * (centeredImage.getYMax()/2) + yp
                ymin = testScale *(-1*centeredImage.getYMin()/2) + yp

                #first assemble a list of detectors which have any hope
                #of overlapping the test image
                viableDetectors = []
                for dd in self.detectors:
                    xOverLaps = False
                    if xmax > dd.xMin and xmax < dd.xMax:
                        xOverLaps = True
                    elif xmin > dd.xMin and xmin < dd.xMax:
                        xOverLaps = True
                    elif xmin < dd.xMin and xmax > dd.xMax:
                        xOverLaps = True

                    yOverLaps = False
                    if ymax > dd.yMin and ymax < dd.yMax:
                        yOverLaps = True
                    elif ymin > dd.yMin and ymin < dd.yMax:
                        yOverLaps = True
                    elif ymin < dd.yMin and ymax > dd.yMax:
                        yOverLaps = True

                    if xOverLaps and yOverLaps and dd not in outputList:
                        viableDetectors.append(dd)


                if len(viableDetectors)>0:

                    #Find the pixels that have a flux greater than 0.001 times the flux of
                    #the central pixel (remember that the object is centered on the test image)
                    maxPixel = centeredImage(centeredImage.getXMax()/2, centeredImage.getYMax()/2)
                    activePixels = numpy.where(centeredImage.array>maxPixel*0.001)

                    #Find the bounds of those active pixels in pixel coordinates
                    xmin = testScale * (activePixels[0].min() - centeredImage.getXMax()/2) + xp
                    xmax = testScale * (activePixels[0].max() - centeredImage.getXMax()/2) + xp
                    ymin = testScale * (activePixels[1].min() - centeredImage.getYMax()/2) + yp
                    ymax = testScale * (activePixels[1].max() - centeredImage.getYMax()/2) + yp

                    #find all of the detectors that overlap with the bounds of the active pixels.
                    for dd in viableDetectors:
                        xOverLaps = False
                        if xmax > dd.xMin and xmax < dd.xMax:
                            xOverLaps = True
                        elif xmin > dd.xMin and xmin < dd.xMax:
                            xOverLaps = True
                        elif xmin < dd.xMin and xmax > dd.xMax:
                            xOverLaps = True

                        yOverLaps = False
                        if ymax > dd.yMin and ymax < dd.yMax:
                            yOverLaps = True
                        elif ymin > dd.yMin and ymin < dd.yMax:
                            yOverLaps = True
                        elif ymin < dd.yMin and ymax > dd.yMax:
                            yOverLaps = True

                        #specifically test that these overlapping detectors do contain active pixels
                        if xOverLaps and yOverLaps:
                            if self._doesObjectImpingeOnDetector(xPupil=xp - centeredImage.getXMax()*testScale/2.0,
                                                                 yPupil=yp - centeredImage.getYMax()*testScale/2.0,
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

    def drawObject(self, galSimType=None, sed=None, ra=None, dec=None, xPupil=None,
                   yPupil=None, halfLightRadius=None, minorAxis=None, majorAxis=None,
                   positionAngle=None, sindex=None):
        """
        Draw an astronomical object on all of the relevant FITS files.

        @param [in] galSimType is a string, either 'pointSource' or 'sersic' denoting the shape of the object

        @param [in] sed is the SED of the object (an instantiation of the Sed class defined in
        sims_photUtils/../../Sed.py

        @param [in] ra is the observed RA coordinate of the object in radians

        @param [in] dec is the observed Dec coordinate of the object in radians

        @param [in] xPupil is the x pupil coordinate of the object in radians

        @param [in] yPupil is the y pupil coordinate of the object in radians

        @param [in] halfLightRadius is the halfLightRadius of the object in radians

        @param [in] minorAxis is the semi-minor axis of the object in radians

        @param [in] majorAxis is the semi-major axis of the object in radians

        @param [in] positionAngle is the position angle of the object in radians

        @param [in] sindex is the sersic index of the object

        @param [out] outputString is a string denoting which detectors the astronomical
        object illumines, suitable for output in the GalSim InstanceCatalog
        """

        #find the detectors which the astronomical object illumines
        outputString, \
        detectorList, \
        centeredObjDict = self.findAllDetectors(galSimType=galSimType,
                                                xPupil=xPupil, yPupil=yPupil,
                                                ra=ra, dec=dec,
                                                halfLightRadius=halfLightRadius,
                                                minorAxis=minorAxis, majorAxis=majorAxis,
                                                positionAngle=positionAngle, sindex=sindex)

        if sed is None or len(detectorList) == 0:
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
                                                                              seeing=self.obs_metadata.seeing[bandpassName],
                                                                              photParams=detector.photParams)

        xp = arcsecFromRadians(xPupil)
        yp = arcsecFromRadians(yPupil)
        spectrum = galsim.SED(spec = lambda ll: numpy.interp(ll, sed.wavelen, sed.flambda),
                              flux_type='flambda')

        for bandpassName in self.bandpasses:

            #create a new object if one has not already been created or if the PSF is wavelength
            #dependent (in which case, each filter is going to need its own initialized object)
            centeredObj = centeredObjDict[bandpassName]
            if centeredObj is None:
                return outputString

            for detector in detectorList:

                name = self._getFileName(detector=detector, bandpassName=bandpassName)

                xPix, yPix = calculatePixelCoordinates(xPupil=numpy.array([xPupil]), yPupil=numpy.array([yPupil]),
                                                       chipNames=[detector.name],
                                                       camera=detector.afwCamera,
                                                       obs_metadata=detector.obs_metadata,
                                                       epoch=detector.epoch)

                obj = centeredObj.copy()

                #convolve the object's shape profile with the spectrum
                obj = obj*spectrum
                localImage = self.blankImage(detector=detector)
                localImage = obj.drawImage(bandpass=self.bandpasses[bandpassName], wcs=detector.wcs,
                                           method='phot', gain=detector.photParams.gain, image=localImage,
                                           offset=galsim.PositionD(xPix[0]-detector.xCenterPix, yPix[0]-detector.yCenterPix),
                                           rng=self._rng)

                self.detectorImages[name] += localImage

        return outputString

    def drawPointSource(self, xPupil=None, yPupil=None, bandpass=None):
        """
        Draw an image of a point source.

        @param [in] xPupil is the x pupil coordinate of the object in arc seconds

        @param [in] yPupil is the y pupil coordinate of the objec tin arc seconds

        @param [in] bandpass is an instantiation of the galsim.Bandpass class characterizing
        the bandpass over which we are integrating (in case the PSF is wavelength dependent)
        """

        if self.PSF is None:
            raise RuntimeError("Cannot draw a point source in GalSim without a PSF")

        return self.PSF.applyPSF(xPupil=xPupil, yPupil=yPupil, bandpass=bandpass)

    def drawSersic(self, xPupil=None, yPupil=None, sindex=None, minorAxis=None,
                   majorAxis=None, positionAngle=None, halfLightRadius=None, bandpass=None):
        """
        Draw the image of a Sersic profile.

        @param [in] xPupil is the x pupil coordinate of the object in arc seconds

        @param [in] yPupil is the y pupil coordinate of the object in arc seconds

        @param [in] sindex is the Sersic index of the object

        @param [in] minorAxis is the semi-minor axis of the object in any units (we only care
        about the ratio of the semi-minor to semi-major axes)

        @param [in] majorAxis is the semi-major axis of the object in the same units
        as minorAxis

        @param [in] halfLightRadius is the half light radius of the object in arc seconds

        @param [in] bandpass is an instantiation of the galsim.Bandpass class characterizing
        the bandpass over which we are integrating (in case the PSF is wavelength dependent)
        """

        #create a Sersic profile
        centeredObj = galsim.Sersic(n=float(sindex), half_light_radius=float(halfLightRadius))

        #turn the Sersic profile into an ellipse
        centeredObj = centeredObj.shear(q=minorAxis/majorAxis, beta=positionAngle*galsim.radians)
        if self.PSF is not None:
            centeredObj = self.PSF.applyPSF(xPupil=xPupil, yPupil=yPupil, obj=centeredObj,
                                            bandpass=bandpass)

        return centeredObj

    def createCenteredObject(self, galSimType=None, xPupil=None, yPupil=None, bandpassName=None, sindex=None,
                             halfLightRadius=None, positionAngle=None, minorAxis=None, majorAxis=None):

        """
        Create a centered GalSim Object (i.e. if we were just to draw this object as an image,
        the object would be centered on the frame)

        @param [in] galSimType is a string denoting the profile of the object (e.g. 'sersic' or 'pointSource')
        If this is a type we have not yet implemented, this method will return None

        @param [in] xPupil is the x pupil coordinate of the object in radians

        @param [in] yPupil is the y pupil coordinate of the object in radians

        @param [in] bandpassName is the tag indicating the bandpass (i.e. 'u', 'g', 'r', 'i', 'z', or 'y')

        @param [in] sindex is the Sersic index of the object

        @param [in] halfLightRadius is the half light radius of the object in radians

        @param [in] positionAngle is the position angle of the object in radians

        @param [in] minorAxis is the semi-minor axis of the object in radians

        @param [in] majorAxis is the semi-major axis of the object in radians

        @param [out] a GalSim Object suitable to base an image off of (but centered on the frame)

        Note: parameters that obviously only apply to Sersic profiles will be ignored in the case
        of point sources
        """

        xp = arcsecFromRadians(xPupil)
        yp = arcsecFromRadians(yPupil)
        hlr = arcsecFromRadians(halfLightRadius)
        if galSimType == 'sersic':
            centeredObj = self.drawSersic(xPupil=xp, yPupil=yp,
                                          bandpass=self.bandpasses[bandpassName],
                                          sindex=sindex, halfLightRadius=hlr,
                                          positionAngle=positionAngle,
                                          minorAxis=minorAxis, majorAxis=majorAxis)
        elif galSimType == 'pointSource':
            centeredObj = self.drawPointSource(xPupil=xp, yPupil=yp,
                                               bandpass=self.bandpasses[bandpassName])
        else:
            print "Apologies: the GalSimInterpreter does not yet have a method to draw "
            print objectParams['galSimType']
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

