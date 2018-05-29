"""
This file defines the following classes:

GalSimInterpreter -- a class which takes objects passed by a GalSim Instance Catalog
(see galSimCatalogs.py) and uses GalSim to write them to FITS images.

GalSimDetector -- a class which stored information about a detector in a way that
GalSimInterpreter expects.
"""
from __future__ import print_function

import math
from builtins import object
import os
import pickle
import tempfile
import numpy as np
import astropy
import galsim
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.utils import radiansFromArcsec
from lsst.sims.GalSimInterface import make_galsim_detector, SNRdocumentPSF, \
    Kolmogorov_and_Gaussian_PSF

__all__ = ["make_gs_interpreter", "GalSimInterpreter", "GalSimSiliconInterpeter"]


def make_gs_interpreter(obs_md, detectors, bandpassDict, noiseWrapper,
                        epoch=None, seed=None, apply_sensor_model=False):
    gs_interpreter \
        = GalSimSiliconInterpeter if apply_sensor_model else GalSimInterpreter

    return gs_interpreter(obs_metadata=obs_md, detectors=detectors,
                          bandpassDict=bandpassDict, noiseWrapper=noiseWrapper,
                          epoch=epoch, seed=seed)


class GalSimInterpreter(object):
    """
    This is the class which actually takes the objects contained in the GalSim
    InstanceCatalog and converts them into FITS images.
    """

    def __init__(self, obs_metadata=None, detectors=None,
                 bandpassDict=None, noiseWrapper=None,
                 epoch=None, seed=None):

        """
        @param [in] obs_metadata is an instantiation of the ObservationMetaData class which
        carries data about this particular observation (telescope site and pointing information)

        @param [in] detectors is a list of GalSimDetectors for which we are drawing FITS images

        @param [in] bandpassDict is a BandpassDict containing all of the bandpasses for which we are
        generating images

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

        self.detectorImages = {}  # this dict will contain the FITS images (as GalSim images)
        self.bandpassDict = bandpassDict
        self.blankImageCache = {}  # this dict will cache blank images associated with specific detectors.
                                   # It turns out that calling the image's constructor is more
                                   # time-consuming than returning a deep copy
        self.checkpoint_file = None
        self.drawn_objects = set()
        self.nobj_checkpoint = 1000
        self._observatory = None

        self.centroid_base_name = None
        self.centroid_handles = {}  # This dict will contain the file handles for each
                                    # centroid file where sources are found.
        self.centroid_list = []  # This is a list of the centroid objects which
                                 # will be written to the file.

    def setPSF(self, PSF=None):
        """
        Set the PSF wrapper for this GalSimInterpreter

        @param [in] PSF is an instantiation of a class which inherits from PSFbase and defines _getPSF()
        """
        self.PSF = PSF

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

        xPupilList = radiansFromArcsec(np.array([xPupil + ix*imgScale for ix in nonZeroPixels[0]]))
        yPupilList = radiansFromArcsec(np.array([yPupil + iy*imgScale for iy in nonZeroPixels[1]]))

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

        @param [out] centeredObj is a GalSim GSObject centered on the chip

        Note: parameters that only apply to Sersic profiles will be ignored in the case of
        pointSources, etc.
        """

        outputString = ''
        outputList = []
        centeredObj = None
        testScale = 0.1

        # create a GalSim Object centered on the chip.
        centeredObj = self.createCenteredObject(gsObject)

        if centeredObj is None:
            return

        # 4 March 2015
        # create a test image of the object to compare against the pixel
        # domains of each detector.  Use photon shooting rather than real space integration
        # for reasons of speed.  A flux of 1000 photons ought to be enough to plot the true
        # extent of the object, but this is just a guess.
        centeredImage = centeredObj.drawImage(scale=testScale, method='phot', n_photons=1000, rng=self._rng)
        xmax = testScale*(centeredImage.xmax/2) + gsObject.xPupilArcsec
        xmin = testScale*(-1*centeredImage.xmax/2) + gsObject.xPupilArcsec
        ymax = testScale*(centeredImage.ymax/2) + gsObject.yPupilArcsec
        ymin = testScale*(-1*centeredImage.ymin/2) + gsObject.yPupilArcsec

        # first assemble a list of detectors which have any hope
        # of overlapping the test image
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

        if len(viableDetectors) > 0:

            # Find the pixels that have a flux greater than 0.001 times the flux of
            # the central pixel (remember that the object is centered on the test image)
            maxPixel = centeredImage(centeredImage.xmax/2, centeredImage.ymax/2)
            activePixels = np.where(centeredImage.array > maxPixel*0.001)

            # Find the bounds of those active pixels in pixel coordinates
            xmin = testScale * (activePixels[0].min() - centeredImage.xmax/2) + gsObject.xPupilArcsec
            xmax = testScale * (activePixels[0].max() - centeredImage.xmax/2) + gsObject.xPupilArcsec
            ymin = testScale * (activePixels[1].min() - centeredImage.ymax/2) + gsObject.yPupilArcsec
            ymax = testScale * (activePixels[1].max() - centeredImage.ymax/2) + gsObject.yPupilArcsec

            # find all of the detectors that overlap with the bounds of the active pixels.
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

                # specifically test that these overlapping detectors do contain active pixels
                if xOverLaps and yOverLaps:
                    if self._doesObjectImpingeOnDetector(xPupil=gsObject.xPupilArcsec -
                                                                centeredImage.xmax*testScale/2.0,
                                                         yPupil=gsObject.yPupilArcsec -
                                                                centeredImage.ymax*testScale/2.0,
                                                         detector=dd, imgScale=centeredImage.scale,
                                                         nonZeroPixels=activePixels):

                        if outputString != '':
                            outputString += '//'
                        outputString += dd.name
                        outputList.append(dd)

        if outputString == '':
            outputString = None

        return outputString, outputList, centeredObj

    def blankImage(self, detector=None):
        """
        Draw a blank image associated with a specific detector.  The image will have the correct size
        for the given detector.

        param [in] detector is an instantiation of GalSimDetector
        """

        # in order to speed up the code (by a factor of ~2), this method
        # only draws a new blank image the first time it is called on a
        # given detector.  It then caches the blank images it has drawn and
        # uses GalSim's copy() method to return copies of cached blank images
        # whenever they are called for again.

        if detector.name in self.blankImageCache:
            return self.blankImageCache[detector.name].copy()
        else:
            image = galsim.Image(detector.xMaxPix-detector.xMinPix+1, detector.yMaxPix-detector.yMinPix+1,
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

        # find the detectors which the astronomical object illumines
        outputString, \
        detectorList, \
        centeredObj = self.findAllDetectors(gsObject)

        if len(detectorList) == 0:
            # there is nothing to draw
            return outputString

        self._addNoiseAndBackground(detectorList)

        for bandpassName in self.bandpassDict:

            # create a new object if one has not already been created or if the PSF is wavelength
            # dependent (in which case, each filter is going to need its own initialized object)
            if centeredObj is None:
                return outputString

            for detector in detectorList:

                name = self._getFileName(detector=detector, bandpassName=bandpassName)

                xPix, yPix = detector.camera_wrapper.pixelCoordsFromPupilCoords(gsObject.xPupilRadians,
                                                                                gsObject.yPupilRadians,
                                                                                detector.name,
                                                                                self.obs_metadata)

                obj = centeredObj

                # convolve the object's shape profile with the spectrum
                obj = obj.withFlux(gsObject.flux(bandpassName))

                self.detectorImages[name] = obj.drawImage(method='phot',
                                                          gain=detector.photParams.gain,
                                                          offset=galsim.PositionD(xPix-detector.xCenterPix,
                                                                                  yPix-detector.yCenterPix),
                                                          rng=self._rng,
                                                          maxN=int(1e6),
                                                          image=self.detectorImages[name],
                                                          add_to_image=True)

                # If we are writing centroid files, store the entry.
                if self.centroid_base_name is not None:
                    centroid_tuple = (detector.fileName, bandpassName, gsObject.uniqueId,
                                      gsObject.flux(bandpassName), xPix, yPix)
                    self.centroid_list.append(centroid_tuple)

        self.drawn_objects.add(gsObject.uniqueId)
        self.write_checkpoint()
        return outputString

    def _addNoiseAndBackground(self, detectorList):
        """
        Go through the list of detector/bandpass combinations and
        initialize all of the FITS files we will need (if they have
        not already been initialized)
        """
        for detector in detectorList:
            for bandpassName in self.bandpassDict:
                name = self._getFileName(detector=detector, bandpassName=bandpassName)
                if name not in self.detectorImages:
                    self.detectorImages[name] = self.blankImage(detector=detector)
                    if self.noiseWrapper is not None:
                        # Add sky background and noise to the image
                        self.detectorImages[name] = \
                            self.noiseWrapper.addNoiseAndBackground(self.detectorImages[name],
                                                                    bandpass=self.bandpassDict[bandpassName],
                                                                    m5=self.obs_metadata.m5[bandpassName],
                                                                    FWHMeff=self.
                                                                    obs_metadata.seeing[bandpassName],
                                                                    photParams=detector.photParams,
                                                                    detector=detector)


    def drawPointSource(self, gsObject, psf=None):
        """
        Draw an image of a point source.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then use self.PSF.
        """
        if psf is None:
            if self.PSF is None:
                raise RuntimeError("Cannot draw a point source in GalSim without a PSF")
            psf = self.PSF

        return psf.applyPSF(xPupil=gsObject.xPupilArcsec, yPupil=gsObject.yPupilArcsec)

    def drawSersic(self, gsObject, psf=None):
        """
        Draw the image of a Sersic profile.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then use self.PSF.
        """

        if psf is None:
            psf = self.PSF

        # create a Sersic profile
        centeredObj = galsim.Sersic(n=float(gsObject.sindex),
                                    half_light_radius=float(gsObject.halfLightRadiusArcsec))

        # Turn the Sersic profile into an ellipse
        centeredObj = centeredObj.shear(q=gsObject.minorAxisRadians/gsObject.majorAxisRadians,
                                        beta=(0.5*np.pi+gsObject.positionAngleRadians)*galsim.radians)

        # Apply weak lensing distortion.
        centeredObj = centeredObj.lens(gsObject.g1, gsObject.g2, gsObject.mu)

        # Apply the PSF.
        if psf is not None:
            centeredObj = psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                                       yPupil=gsObject.yPupilArcsec,
                                       obj=centeredObj)

        return centeredObj

    def drawRandomWalk(self, gsObject, psf=None):
        """
        Draw the image of a RandomWalk light profile. In orider to allow for
        reproducibility, the specific realisation of the random walk is seeded
        by the object unique identifier, if provided.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then use self.PSF.
        """
        if psf is None:
            psf = self.PSF
        # Seeds the random walk with the object id if available
        if gsObject.uniqueId is None:
            rng = None
        else:
            rng = galsim.BaseDeviate(int(gsObject.uniqueId))

        # Create the RandomWalk profile
        centeredObj = galsim.RandomWalk(npoints=int(gsObject.npoints),
                                        half_light_radius=float(gsObject.halfLightRadiusArcsec),
                                        rng=rng)

        # Apply intrinsic ellipticity to the profile
        centeredObj = centeredObj.shear(q=gsObject.minorAxisRadians/gsObject.majorAxisRadians,
                                        beta=(0.5*np.pi+gsObject.positionAngleRadians)*galsim.radians)

        # Apply weak lensing distortion.
        centeredObj = centeredObj.lens(gsObject.g1, gsObject.g2, gsObject.mu)

        # Apply the PSF.
        if psf is not None:
            centeredObj = psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                                       yPupil=gsObject.yPupilArcsec,
                                       obj=centeredObj)

        return centeredObj

    def createCenteredObject(self, gsObject, psf=None):
        """
        Create a centered GalSim Object (i.e. if we were just to draw this object as an image,
        the object would be centered on the frame)

        @param [in] gsObject is an instantiation of the GalSimCelestialObject class
        carrying information about the object whose image is to be drawn

        Note: parameters that obviously only apply to Sersic profiles will be ignored in the case
        of point sources
        """

        if gsObject.galSimType == 'sersic':
            centeredObj = self.drawSersic(gsObject, psf=psf)

        elif gsObject.galSimType == 'pointSource':
            centeredObj = self.drawPointSource(gsObject, psf=psf)

        elif gsObject.galSimType == 'RandomWalk':
            centeredObj = self.drawRandomWalk(gsObject, psf=psf)

        else:
            print("Apologies: the GalSimInterpreter does not yet have a method to draw ")
            print(gsObject.galSimType)
            print(" objects\n")
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

    def open_centroid_file(self, centroid_name):
        """
        Open a centroid file.  This file will have one line per-object and the
        it will be labeled with the objectID and then followed by the average X
        Y position of the photons from the object. Either the true photon
        position or the average of the pixelated electrons collected on a finite
        sensor can be chosen.
        """

        visitID = self.obs_metadata.OpsimMetaData['obshistID']
        file_name = self.centroid_base_name + str(visitID) + '_' + centroid_name + '.txt'

        self.centroid_handles[centroid_name] = open(file_name, 'w')
        self.centroid_handles[centroid_name].write('{:15} {:>15} {:>10} {:>10}\n'.
                                                   format('SourceID', 'Flux', 'xPix', 'yPix'))

    def _writeObjectToCentroidFile(self, detector_name, bandpass_name, uniqueId, flux, xPix, yPix):
        """
        Write the flux and the the object position on the sensor for this object
        into a centroid file.  First check if a centroid file exists for this
        detector and, if it doesn't create it.

        @param [in] detectorName is the name of the sensor the gsObject falls on.

        @param [in] bandpassName is the name of the filter used in this exposure.

        @param [in] UniqueID is the Unique ID of the gsObject.

        @param [in] flux is the calculated flux for the gsObject in the given bandPass.
        """

        centroid_name = detector_name + '_' + bandpass_name

        # If we haven't seen this sensor before open a centroid file for it.
        if centroid_name not in self.centroid_handles:
            self.open_centroid_file(centroid_name)

        # Write the object to the file
        self.centroid_handles[centroid_name].write('{:<15d} {:15.5f} {:10.2f} {:10.2f}\n'.
                                                   format(uniqueId, flux, xPix, yPix))

    def write_centroid_files(self):
        """
        Write the centroid data structure out to the files.

        This function loops over the entries in the centroid list and
        then sends them each to be writen to a file. The
        _writeObjectToCentroidFile will decide how to put them in files.

        After writing the files are closed.
        """
        # Loop over entries
        for centroid_tuple in self.centroid_list:
            self._writeObjectToCentroidFile(*centroid_tuple)

        # Now close the centroid files.
        for name in self.centroid_handles:
            self.centroid_handles[name].close()

    def write_checkpoint(self, force=False):
        """
        Write a pickle file of detector images packaged with the
        objects that have been drawn. By default, write the checkpoint
        every self.nobj_checkpoint objects.
        """
        if self.checkpoint_file is None:
            return
        if force or len(self.drawn_objects) % self.nobj_checkpoint == 0:
            # The galsim.Images in self.detectorImages cannot be
            # pickled because they contain references to unpickleable
            # afw objects, so just save the array data and rebuild
            # the galsim.Images from scratch, given the detector name.
            images = {key: value.array for key, value
                      in self.detectorImages.items()}
            image_state = dict(images=images,
                               rng=self._rng,
                               drawn_objects=self.drawn_objects,
                               centroid_objects=self.centroid_list)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                             dir='.') as tmp:
                pickle.dump(image_state, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.rename(tmp.name, self.checkpoint_file)

    def restore_checkpoint(self, camera_wrapper, phot_params, obs_metadata,
                           epoch=2000.0):
        """
        Restore self.detectorImages, self._rng, and self.drawn_objects states
        from the checkpoint file.

        Parameters
        ----------
        camera_wrapper: lsst.sims.GalSimInterface.GalSimCameraWrapper
            An object representing the camera being simulated

        phot_params: lsst.sims.photUtils.PhotometricParameters
            An object containing the physical parameters representing
            the photometric properties of the system

        obs_metadata: lsst.sims.utils.ObservationMetaData
            Characterizing the pointing of the telescope

        epoch: float
            Representing the Julian epoch against which RA, Dec are
            reckoned (default = 2000)
        """
        if (self.checkpoint_file is None
            or not os.path.isfile(self.checkpoint_file)):
            return
        with open(self.checkpoint_file, 'rb') as input_:
            image_state = pickle.load(input_)
            images = image_state['images']
            for key in images:
                # Unmangle the detector name.
                detname = "R:{},{} S:{},{}".format(*tuple(key[1:3] + key[5:7]))
                # Create the galsim.Image from scratch as a blank image and
                # set the pixel data from the persisted image data array.
                detector = make_galsim_detector(camera_wrapper, detname,
                                                phot_params, obs_metadata,
                                                epoch=epoch)
                self.detectorImages[key] = self.blankImage(detector=detector)
                self.detectorImages[key] += image_state['images'][key]
            self._rng = image_state['rng']
            self.drawn_objects = image_state['drawn_objects']
            self.centroid_list = image_state['centroid_objects']

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        obs_location = astropy.coordinates.EarthLocation.from_geodetic(
            self.observatory.getLongitude().asDegrees(),
            self.observatory.getLatitude().asDegrees(),
            self.observatory.getElevation())
        time = astropy.time.Time(mjd, format='mjd', location=obs_location)
        ra_val = ra if ra > 0 else 360. + ra
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        last = last if last > 0 else 360. + last
        ha =  last - ra_val
        return ha if ha > 0 else 360 + ha

    @property
    def observatory(self):
        if self._observatory is None:
            self._observatory \
                = LsstSimMapper().MakeRawVisitInfoClass().observatory
        return self._observatory


class GalSimSiliconInterpeter(GalSimInterpreter):
    """
    This subclass of GalSimInterpreter applies the Silicon sensor
    model to the drawn objects.
    """
    def __init__(self, obs_metadata=None, detectors=None, bandpassDict=None,
                 noiseWrapper=None, epoch=None, seed=None):
        super(GalSimSiliconInterpeter, self)\
            .__init__(obs_metadata=obs_metadata, detectors=detectors,
                      bandpassDict=bandpassDict, noiseWrapper=noiseWrapper,
                      epoch=epoch, seed=seed)

        self.sky_bg_per_pixel = None

        # Create a PSF that's fast to evaluate for the postage stamp
        # size calculation for extended objects in .getStampBounds.
        FWHMgeom = obs_metadata.OpsimMetaData['FWHMgeom']
        self._double_gaussian_psf = SNRdocumentPSF(FWHMgeom)

        # Save the parameters needed to create a Kolmogorov PSF for a
        # custom value of gsparams.folding_threshold.  That PSF will
        # to be used in the .getStampBounds function for bright stars.
        altRad = np.radians(obs_metadata.OpsimMetaData['altitude'])
        self._airmass = 1.0/np.sqrt(1.0-0.96*(np.sin(0.5*np.pi-altRad))**2)
        self._rawSeeing = obs_metadata.OpsimMetaData['rawSeeing']
        self._band = obs_metadata.bandpass

        # Save the default folding threshold for determining when to recompute
        # the PSF for bright point sources.
        self._ft_default = galsim.GSParams().folding_threshold

    def drawObject(self, gsObject):
        """
        Draw an astronomical object on all of the relevant FITS files.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject
        class carrying all of the information for the object whose image
        is to be drawn

        @param [out] outputString is a string denoting which detectors the astronomical
        object illumines, suitable for output in the GalSim InstanceCatalog
        """

        # find the detectors which the astronomical object illumines
        outputString, \
        detectorList, \
        centeredObj = self.findAllDetectors(gsObject)

        if len(detectorList) == 0:
            # there is nothing to draw
            return outputString

        self._addNoiseAndBackground(detectorList)

        # Create a surface operation to sample incident angles and a
        # galsim.SED object for sampling the wavelengths of the
        # incident photons.
        fratio = 1.234  # From https://www.lsst.org/scientists/keynumbers
        obscuration = 0.606  # (8.4**2 - 6.68**2)**0.5 / 8.4
        angles = galsim.FRatioAngles(fratio, obscuration, self._rng)

        sed_lut = galsim.LookupTable(x=gsObject.sed.wavelen,
                                     f=gsObject.sed.flambda)
        gs_sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda',
                            redshift=0.)

        for bandpassName in self.bandpassDict:
            # create a new object if one has not already been created
            # or if the PSF is wavelength dependent (in which case,
            # each filter is going to need its own initialized object)
            if centeredObj is None:
                return outputString

            bandpass = self.bandpassDict[bandpassName]
            index = np.where(bandpass.sb != 0)
            bp_lut = galsim.LookupTable(x=bandpass.wavelen[index],
                                        f=bandpass.sb[index])
            gs_bandpass = galsim.Bandpass(bp_lut, wave_type='nm')
            waves = galsim.WavelengthSampler(sed=gs_sed, bandpass=gs_bandpass,
                                             rng=self._rng)

            # Set the object flux.
            flux = gsObject.flux(bandpassName)
            obj = centeredObj.withFlux(flux)

            for detector in detectorList:

                name = self._getFileName(detector=detector,
                                         bandpassName=bandpassName)

                xPix, yPix = detector.camera_wrapper\
                   .pixelCoordsFromPupilCoords(gsObject.xPupilRadians,
                                               gsObject.yPupilRadians,
                                               chipName=detector.name,
                                               obs_metadata=self.obs_metadata)

                sensor = galsim.SiliconSensor(rng=self._rng,
                                              treering_center=detector.tree_rings.center,
                                              treering_func=detector.tree_rings.func,
                                              transpose=True)
                surface_ops = [waves, angles]

                # Desired position to draw the object.
                image_pos = galsim.PositionD(xPix, yPix)

                # Find a postage stamp region to draw onto.  Use (sky
                # noise)/3. as the nominal minimum surface brightness
                # for rendering an extended object.
                keep_sb_level = np.sqrt(self.sky_bg_per_pixel)/3.
                bounds = self.getStampBounds(gsObject, flux, image_pos,
                                             keep_sb_level, 3*keep_sb_level)

                # Ensure the bounds of the postage stamp lie within the image.
                bounds = bounds & self.detectorImages[name].bounds

                # offset is relative to the "true" center of the postage stamp.
                offset = image_pos - bounds.true_center

                if bounds.area() > 0:
                    obj.drawImage(method='phot',
                                  offset=offset,
                                  rng=self._rng,
                                  maxN=int(1e6),
                                  image=self.detectorImages[name][bounds],
                                  sensor=sensor,
                                  surface_ops=surface_ops,
                                  add_to_image=True,
                                  gain=detector.photParams.gain)

                    # If we are writing centroid files,store the entry.
                    if self.centroid_base_name is not None:
                        centroid_tuple = (detector.fileName, bandpassName, gsObject.uniqueId,
                                          gsObject.flux(bandpassName), xPix, yPix)
                        self.centroid_list.append(centroid_tuple)

        self.drawn_objects.add(gsObject.uniqueId)
        self.write_checkpoint()
        return outputString

    def getStampBounds(self, gsObject, flux, image_pos, keep_sb_level,
                       large_object_sb_level, Nmax=1400, pixel_scale=0.2):
        """
        Get the postage stamp bounds for drawing an object within the stamp
        to include the specified minimum surface brightness.  Use the
        folding_threshold criterion for point source objects.  For
        extended objects, use the getGoodPhotImageSize function, where
        if the initial stamp is too large (> Nmax**2 ~ 1GB of RSS
        memory for a 72 vertex/pixel sensor model), use the relaxed
        surface brightness level for large objects.

        Parameters
        ----------
        gsObject: GalSimCelestialObject
            This contains the information needed to construct a
            galsim.GSObject convolved with the desired PSF.
        flux: float
            The flux of the object in e-.
        keep_sb_level: float
            The minimum surface brightness (photons/pixel) out to which to
            extend the postage stamp, e.g., a value of
            sqrt(sky_bg_per_pixel)/3 would be 1/3 the Poisson noise
            per pixel from the sky background.
        large_object_sb_level: float
            Surface brightness level to use for large/bright objects that
            would otherwise yield stamps with more than Nmax**2 pixels.
        Nmax: int [1400]
            The largest stamp size to consider at the nominal keep_sb_level.
            1400**2*72*8/1024**3 = 1GB.
        pixel_scale: float [0.2]
            The CCD pixel scale in arcsec.

        Returns
        -------
        galsim.BoundsI: The postage stamp bounds.

        """
        if gsObject.galSimType.lower() == "pointsource":
            # For bright stars, set the folding threshold for the
            # stamp size calculation.  Use a
            # Kolmogorov_and_Gaussian_PSF since it is faster to
            # evaluate than an AtmosphericPSF.
            folding_threshold = self.sky_bg_per_pixel/flux
            if folding_threshold >= self._ft_default:
                gsparams = None
            else:
                gsparams = galsim.GSParams(folding_threshold=folding_threshold)
            psf = Kolmogorov_and_Gaussian_PSF(airmass=self._airmass,
                                              rawSeeing=self._rawSeeing,
                                              band=self._band,
                                              gsparams=gsparams)
            obj = self.drawPointSource(gsObject, psf=psf)
            image_size = obj.getGoodImageSize(pixel_scale)
        else:
            # For extended objects, recreate the object to draw, but
            # convolved with the faster DoubleGaussian PSF.
            obj = self.createCenteredObject(gsObject,
                                            psf=self._double_gaussian_psf)
            obj = obj.withFlux(flux)
            image_size = getGoodPhotImageSize(obj, keep_sb_level,
                                              pixel_scale=pixel_scale)
            if image_size > Nmax:
                image_size = getGoodPhotImageSize(obj, large_object_sb_level,
                                                  pixel_scale=pixel_scale)
                image_size = max(image_size, Nmax)

        # Create the bounds object centered on the desired location.
        xmin = int(math.floor(image_pos.x) - image_size/2)
        xmax = int(math.ceil(image_pos.x) + image_size/2)
        ymin = int(math.floor(image_pos.y) - image_size/2)
        ymax = int(math.ceil(image_pos.y) + image_size/2)

        return galsim.BoundsI(xmin, xmax, ymin, ymax)


def getGoodPhotImageSize(obj, keep_sb_level, pixel_scale=0.2):
    """
    Get a postage stamp size (appropriate for photon-shooting) given a
    minimum surface brightness in photons/pixel out to which to
    extend the stamp region.

    Parameters
    ----------
    obj: galsim.GSObject
        The GalSim object for which we will call .drawImage.
    keep_sb_level: float
        The minimum surface brightness (photons/pixel) out to which to
        extend the postage stamp, e.g., a value of
        sqrt(sky_bg_per_pixel)/3 would be 1/3 the Poisson noise
        per pixel from the sky background.
    pixel_scale: float [0.2]
        The CCD pixel scale in arcsec.

    Returns
    -------
    int: The length N of the desired NxN postage stamp.

    Notes
    -----
    Use of this function should be avoided with PSF implementations that
    are costly to evaluate.  A roughly equivalent DoubleGaussian
    could be used as a proxy.

    This function was originally written by Mike Jarvis.
    """
    # The factor by which to adjust N in each step.
    factor = 1.1

    # Start with the normal image size from GalSim
    N = obj.getGoodImageSize(pixel_scale)
    #print('N = ',N)

    # This can be too small for bright stars, so increase it in steps until the edges are
    # all below the requested sb level.
    # (Don't go bigger than 4096)
    Nmax = 4096
    while N < Nmax:
        # Check the edges and corners of the current square
        h = N / 2 * pixel_scale
        xvalues = [ obj.xValue(h,0), obj.xValue(-h,0),
                    obj.xValue(0,h), obj.xValue(0,-h),
                    obj.xValue(h,h), obj.xValue(h,-h),
                    obj.xValue(-h,h), obj.xValue(-h,-h) ]
        maxval = np.max(xvalues)
        #print(N, maxval)
        if maxval < keep_sb_level:
            break
        N *= factor

    N = min(N, Nmax)

    # This can be quite huge for Devauc profiles, but we don't actually have much
    # surface brightness way out in the wings.  So cut it back some.
    # (Don't go below 64 though.)
    while N >= 64 * factor:
        # Check the edges and corners of a square smaller by a factor of N.
        h = N / (2 * factor) * pixel_scale
        xvalues = [ obj.xValue(h,0), obj.xValue(-h,0),
                    obj.xValue(0,h), obj.xValue(0,-h),
                    obj.xValue(h,h), obj.xValue(h,-h),
                    obj.xValue(-h,h), obj.xValue(-h,-h) ]
        maxval = np.max(xvalues)
        #print(N, maxval)
        if maxval > keep_sb_level:
            break
        N /= factor

    return int(N)
