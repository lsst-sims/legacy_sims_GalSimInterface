"""
This file defines GalSimBase, which is a daughter of InstanceCatalog designed
to interface with GalSimInterpreter and generate images using GalSim.

It also defines daughter classes of GalSimBase designed for specific
classes of astronomical objects:

GalSimGalaxies
GalSimAgn
GalSimStars
"""

import numpy
import os
import copy
from itertools import izip
import lsst.utils
from lsst.sims.utils import arcsecFromRadians
from lsst.sims.catalogs.definitions import InstanceCatalog, is_null
from lsst.sims.catalogs.decorators import cached
from lsst.sims.catUtils.mixins import CameraCoords, AstrometryGalaxies, AstrometryStars, \
                                      EBVmixin
from lsst.sims.GalSimInterface import GalSimInterpreter, GalSimDetector, GalSimCelestialObject
from lsst.sims.photUtils import Sed, Bandpass, BandpassDict, \
                                PhotometricParameters, LSSTdefaults
import lsst.afw.cameraGeom.testUtils as camTestUtils
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE

__all__ = ["GalSimGalaxies", "GalSimAgn", "GalSimStars"]

class GalSimBase(InstanceCatalog, CameraCoords):
    """
    The catalog classes in this file use the InstanceCatalog infrastructure to construct
    FITS images for each detector-filter combination on a simulated camera.  This is done by
    instantiating the class GalSimInterpreter.  GalSimInterpreter is the class which
    actually generates the FITS images.  As the GalSim InstanceCatalogs are iterated over,
    each object in the catalog is passed to the GalSimInterpeter, which adds the object
    to the appropriate FITS images.  The user can then write the images to disk by calling
    the write_images method in the GalSim InstanceCatalog.

    Objects are passed to the GalSimInterpreter by the get_fitsFiles getter function, which
    adds a column to the InstanceCatalog indicating which detectors' FITS files contain each
    object.

    Note: because each GalSim InstanceCatalog has its own GalSimInterpreter, it means
    that each GalSimInterpreter will only draw FITS images containing one type of object
    (whatever type of object is contained in the GalSim InstanceCatalog).  If the user
    wishes to generate FITS images containing multiple types of object, the method
    copyGalSimInterpreter allows the user to pass the GalSimInterpreter from one
    GalSim InstanceCatalog to another (so, the user could create a GalSim Instance
    Catalog of stars, generate that InstanceCatalog, then create a GalSim InstanceCatalog
    of galaxies, pass the GalSimInterpreter from the star catalog to this new catalog,
    and thus create FITS images that contain both stars and galaxies; see galSimCompoundGenerator.py
    in the examples/ directory of sims_catUtils for an example).

    This class (GalSimBase) is the base class for all GalSim InstanceCatalogs.  Daughter
    classes of this class need to behave like ordinary InstanceCatalog daughter classes
    with the following exceptions:

    1) If they re-define column_outputs, they must be certain to include the column
    'fitsFiles.'  The getter for this column (defined in this class) calls all of the
    GalSim image generation infrastructure

    2) Daughter classes of this class must define a member variable galsim_type that is either
    'sersic' or 'pointSource'.  This variable tells the GalSimInterpreter how to draw the
    object (to allow a different kind of image profile, define a new method in the GalSimInterpreter
    class similar to drawPoinSource and drawSersic)

    3) The variables bandpass_names (a list of the form ['u', 'g', 'r', 'i', 'z', 'y']),
    bandpass_directory, and bandpass_root should be defined to tell the GalSim InstanceCatalog
    where to find the files defining the bandpasses to be used for these FITS files.
    The GalSim InstanceCatalog will look for bandpass files in files with the names

    for bpn in bandpass_names:
        name = self.bandpass_directory+'/'+self.bandpass_root+'_'+bpn+'.dat'

    one should also define the following member variables:

        componentList is a list of files ins banpass_directory containing the response
        curves for the different components of the camera, e.g.
        ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat', 'lens1.dat', 'lens2.dat', 'lens3.dat']

        atomTransmissionName is the name of the file in bandpass_directory that contains the
        atmostpheric transmissivity, e.g. 'atmos_std.dat'

    4) Telescope parameters such as exposure time, area, and gain are stored in the
    GalSim InstanceCatalog member variable photParams, which is an instantiation of
    the class PhotometricParameters defined in sims_photUtils.

    Daughter classes of GalSimBase will generate both FITS images for all of the detectors/filters
    in their corresponding cameras and InstanceCatalogs listing all of the objects
    contained in those images.  The catalog is written using the normal write_catalog()
    method provided for all InstanceClasses.  The FITS files are drawn using the write_images()
    method that is unique to GalSim InstanceCatalogs.  The FITS file will be named something like:

    DetectorName_FilterName.fits

    (a typical LSST fits file might be R_0_0_S_1_0_y.fits)

    Note: If you call write_images() before iterating over the catalog (either by calling
    write_catalog() or using the iterator returned by InstanceCatalog.iter_catalog()),
    you will get empty or incomplete FITS files.  Objects are only added to the GalSimInterpreter
    in the course of iterating over the InstanceCatalog.
    """

    seed = 42

    #This is sort of a hack; it prevents findChipName in coordUtils from dying
    #if an object lands on multiple science chips.
    allow_multiple_chips = True

    #There is no point in writing things to the InstanceCatalog that do not have SEDs and/or
    #do not land on any detectors
    cannot_be_null = ['sedFilepath', 'fitsFiles']

    column_outputs = ['galSimType', 'uniqueId', 'raICRS', 'decICRS',
                      'chipName', 'x_pupil', 'y_pupil', 'sedFilepath',
                      'majorAxis', 'minorAxis', 'sindex', 'halfLightRadius',
                      'positionAngle','fitsFiles']

    transformations = {'raICRS':numpy.degrees,
                       'decICRS':numpy.degrees,
                       'x_pupil':arcsecFromRadians,
                       'y_pupil':arcsecFromRadians,
                       'halfLightRadius':arcsecFromRadians}

    default_formats = {'S':'%s', 'f':'%.9g', 'i':'%i'}

    #This is used as the delimiter because the names of the detectors printed in the fitsFiles
    #column contain both ':' and ','
    delimiter = '; '

    sedDir = lsst.utils.getPackageDir('sims_sed_library')

    bandpassNames = ['u', 'g', 'r', 'i', 'z', 'y']
    bandpassDir = os.path.join(lsst.utils.getPackageDir('throughputs'), 'baseline')
    bandpassRoot = 'filter_'
    componentList = ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat',
                     'lens1.dat', 'lens2.dat', 'lens3.dat']
    atmoTransmissionName = 'atmos_std.dat'

    # allowed_chips is a list of the names of the detectors we actually want to draw.
    # If 'None', then all chips are drawn.
    allowed_chips = None

    #This member variable will define a PSF to convolve with the sources.
    #See the classes PSFbase and DoubleGaussianPSF in
    #galSimUtilities.py for more information
    PSF = None

    #This member variable can store a GalSim noise model instantiation
    #which will be applied to the FITS images when they are created
    noise_and_background = None

    #Stores the gain and readnoise
    photParams = PhotometricParameters()

    #This is just a place holder for the camera object associated with the InstanceCatalog.
    #If you want to assign a different camera, you can do so immediately after instantiating this class
    camera = camTestUtils.CameraWrapper().camera


    uniqueSeds = {} #a cache for un-normalized SED files, so that we do not waste time on I/O

    hasBeenInitialized = False

    galSimInterpreter = None #the GalSimInterpreter instantiation for this catalog

    totalDrawings = 0
    totalObjects = 0

    def _initializeGalSimCatalog(self):
        """
        Initializes an empty list of objects that have already been drawn to FITS images.
        We do not want to accidentally draw an object twice.

        Also initializes the GalSimInterpreter by calling self._initializeGalSimInterpreter()

        Objects are stored based on their uniqueId values.
        """
        self.objectHasBeenDrawn = []
        self._initializeGalSimInterpreter()
        self.hasBeenInitialized = True

    def get_sedFilepath(self):
        """
        Maps the name of the SED as stored in the database to the file stored in
        sims_sed_library
        """
        #copied from the phoSim catalogs
        return numpy.array([self.specFileMap[k] if k in self.specFileMap else None
                         for k in self.column_by_name('sedFilename')])

    def _calcGalSimSed(self, sedName, zz, iAv, iRv, gAv, gRv, norm):
        """
        correct the SED for redshift, dust, etc.  Return an Sed object as defined in
        sims_photUtils/../../Sed.py
        """
        if is_null(sedName):
            return None
        sed = self._getSedCopy(sedName)
        imsimband = Bandpass()
        imsimband.imsimBandpass()
        #normalize the SED
        #Consulting the file sed.py in GalSim/galsim/ it appears that GalSim expects
        #its SEDs to ultimately be in units of ergs/nm so that, when called, they can
        #be converted to photons/nm (see the function __call__() and the assignment of
        #self._rest_photons in the __init__() of galsim's sed.py file).  Thus, we need
        #to read in our SEDs, normalize them, and then multiply by the exposure time
        #and the effective area to get from ergs/s/cm^2/nm to ergs/nm.
        #
        #The gain parameter should convert between photons and ADU (so: it is the
        #traditional definition of "gain" -- electrons per ADU -- multiplied by the
        #quantum efficiency of the detector).  Because we fold the quantum efficiency
        #of the detector into our total_[u,g,r,i,z,y].dat bandpass files
        #(see the readme in the THROUGHPUTS_DIR/baseline/), we only need to multiply
        #by the electrons per ADU gain.
        #
        #We will take these parameters from an instantiation of the PhotometricParameters
        #class (which can be reassigned by defining a daughter class of this class)
        #
        fNorm = sed.calcFluxNorm(norm, imsimband)
        sed.multiplyFluxNorm(fNorm)

        # apply dust extinction (internal)
        if iAv != 0.0 and iRv != 0.0:
            a_int, b_int = sed.setupCCMab()
            sed.addCCMDust(a_int, b_int, A_v=iAv, R_v=iRv)

        # 22 June 2015
        # apply redshift; there is no need to apply the distance modulus from
        # sims/photUtils/CosmologyWrapper; magNorm takes that into account
        # however, magNorm does not take into account cosmological dimming
        if zz != 0.0:
            sed.redshiftSED(zz, dimming=True)

        # apply dust extinction (galactic)
        a_int, b_int = sed.setupCCMab()
        sed.addCCMDust(a_int, b_int, A_v=gAv, R_v=gRv)
        return sed

    def _getSedCopy(self, sedName):
        """
        Return a copy of the requested SED, either from the cached
        version or creating a new one and caching a copy for later
        reuse.
        """
        if sedName in self.uniqueSeds:
            # we have already read in this file; no need to do it again
            sed = Sed(wavelen=self.uniqueSeds[sedName].wavelen,
                      flambda=self.uniqueSeds[sedName].flambda,
                      fnu=self.uniqueSeds[sedName].fnu,
                      name=self.uniqueSeds[sedName].name)
        else:
            # load the SED of the object
            sed = Sed()
            sedFile = os.path.join(self.sedDir, sedName)
            sed.readSED_flambda(sedFile)

            flambdaCopy = copy.deepcopy(sed.flambda)

            #If the SED is zero inside of the bandpass, GalSim raises an error.
            #This sets a minimum flux value of 1.0e-30 so that the SED is never technically
            #zero inside of the bandpass.
            sed.flambda = numpy.array([ff if ff>1.0e-30 else 1.0e-30 for ff in flambdaCopy])
            sed.fnu = None

            #copy the unnormalized file to uniqueSeds so we don't have to read it in again
            sedCopy = Sed(wavelen=sed.wavelen, flambda=sed.flambda,
                          fnu=sed.fnu, name=sed.name)
            self.uniqueSeds[sedName] = sedCopy
        return sed

    def _calculateGalSimSeds(self):
        """
        Apply any physical corrections to the objects' SEDS (redshift them, apply dust, etc.).

        Return a generator that serves up the Sed objects in order.
        """
        actualSEDnames = self.column_by_name('sedFilepath')
        redshift = self.column_by_name('redshift')
        internalAv = self.column_by_name('internalAv')
        internalRv = self.column_by_name('internalRv')
        galacticAv = self.column_by_name('galacticAv')
        galacticRv = self.column_by_name('galacticRv')
        magNorm = self.column_by_name('magNorm')

        return (self._calcGalSimSed(*args) for args in
                zip(actualSEDnames, redshift, internalAv, internalRv,
                    galacticAv, galacticRv, magNorm))

    def get_fitsFiles(self):
        """
        This getter returns a column listing the names of the detectors whose corresponding
        FITS files contain the object in question.  The detector names will be separated by a '//'

        This getter also passes objects to the GalSimInterpreter to actually draw the FITS
        images.
        """
        objectNames = self.column_by_name('uniqueId')
        raICRS = self.column_by_name('raICRS')
        decICRS = self.column_by_name('decICRS')
        xPupil = self.column_by_name('x_pupil')
        yPupil = self.column_by_name('y_pupil')
        halfLight = self.column_by_name('halfLightRadius')
        minorAxis = self.column_by_name('minorAxis')
        majorAxis = self.column_by_name('majorAxis')
        positionAngle = self.column_by_name('positionAngle')
        sindex = self.column_by_name('sindex')

        sedList = self._calculateGalSimSeds()

        if self.hasBeenInitialized is False and len(objectNames)>0:
            #This needs to be here in case, instead of writing the whole catalog with write_catalog(),
            #the user wishes to iterate through the catalog with InstanceCatalog.iter_catalog(),
            #which will not call write_header()
            self._initializeGalSimCatalog()
            if not hasattr(self, 'bandpassDict'):
                raise RuntimeError('ran initializeGalSimCatalog but do not have bandpassDict')

        output = []
        for (name, ra, dec, xp, yp, hlr, minor, major, pa, ss, sn) in \
            izip(objectNames, raICRS, decICRS, xPupil, yPupil, halfLight, \
                minorAxis, majorAxis, positionAngle, sedList, sindex):

            if ss is None or name in self.objectHasBeenDrawn:
                #do not draw objects that have no SED or have already been drawn
                output.append(None)
                if name in self.objectHasBeenDrawn:
                    #15 December 2014
                    #This should probably be an error.  However, something is wrong with
                    #the SQL on fatboy such that it does return the same objects more than
                    #once (at least in the case of stars).  Yusra is currently working to fix
                    #the problem.  Until then, this will just warn you that the same object
                    #appears twice in your catalog and will refrain from drawing it the second
                    #time.
                    print 'Trying to draw %s more than once ' % str(name)

            else:

                self.objectHasBeenDrawn.append(name)

                flux_dict = {}
                for bb in self.bandpassNames:
                    adu = ss.calcADU(self.bandpassDict[bb], self.photParams)
                    flux_dict[bb] = adu*self.photParams.gain

                gsObj = GalSimCelestialObject(self.galsim_type, ss, ra, dec, xp, yp, \
                                              hlr, minor, major, pa, sn, flux_dict)

                #actually draw the object
                detectorsString = self.galSimInterpreter.drawObject(gsObj)

                output.append(detectorsString)

        return numpy.array(output)


    def setPSF(self, PSF):
        """
        Set the PSF of this GalSimCatalog after instantiation.

        @param [in] PSF is an instantiation of a GalSimPSF class.
        """
        self.PSF=PSF
        if self.galSimInterpreter is not None:
            self.galSimInterpreter.setPSF(PSF=PSF)


    def copyGalSimInterpreter(self, otherCatalog):
        """
        Copy the camera, GalSimInterpreter, from another GalSim InstanceCatalog
        so that multiple types of object (stars, AGN, galaxy bulges, galaxy disks, etc.)
        can be drawn on the same FITS files.

        Note: This method does not copy the member variables PSF or noise_and_background
        from one catalog to another.  Those need to be defined in each catalog separately.

        @param [in] otherCatalog is another GalSim InstanceCatalog that already has
        an initialized GalSimInterpreter

        See galSimCompoundGenerator.py in the examples/ directory of sims_catUtils for
        an example of how this is used.
        """
        self.camera = otherCatalog.camera
        self.photParams = otherCatalog.photParams
        self.bandpassDict = otherCatalog.bandpassDict
        self.galSimInterpreter = otherCatalog.galSimInterpreter


    def _initializeGalSimInterpreter(self):
        """
        This method creates the GalSimInterpreter (if it is None)

        This method reads in all of the data about the camera and pass it into
        the GalSimInterpreter.

        This method calls _getBandpasses to construct the paths to
        the files containing the bandpass data.
        """

        if self.galSimInterpreter is None:

            #This list will contain instantiations of the GalSimDetector class
            #(see galSimInterpreter.py), which stores detector information in a way
            #that the GalSimInterpreter will understand
            detectors = []

            for dd in self.camera:
                if self.allowed_chips is None or dd.getName() in self.allowed_chips:
                    cs = dd.makeCameraSys(PUPIL)
                    centerPupil = self.camera.transform(dd.getCenter(FOCAL_PLANE),cs).getPoint()
                    centerPixel = dd.getCenter(PIXELS).getPoint()

                    translationPixel = afwGeom.Point2D(centerPixel.getX()+1, centerPixel.getY()+1)
                    translationPupil = self.camera.transform(
                                            dd.makeCameraPoint(translationPixel, PIXELS), cs).getPoint()

                    plateScale = numpy.sqrt(numpy.power(translationPupil.getX()-centerPupil.getX(),2)+
                                            numpy.power(translationPupil.getY()-centerPupil.getY(),2))/numpy.sqrt(2.0)

                    plateScale = 3600.0*numpy.degrees(plateScale)

                    #make a detector-custom photParams that copies all of the quantities
                    #in the catalog photParams, except the platescale, which is
                    #calculated above
                    params = PhotometricParameters(exptime=self.photParams.exptime,
                                                   nexp=self.photParams.nexp,
                                                   effarea=self.photParams.effarea,
                                                   gain=self.photParams.gain,
                                                   readnoise=self.photParams.readnoise,
                                                   darkcurrent=self.photParams.darkcurrent,
                                                   othernoise=self.photParams.othernoise,
                                                   platescale=plateScale)


                    detector = GalSimDetector(dd, self.camera,
                                              obs_metadata=self.obs_metadata, epoch=self.db_obj.epoch,
                                              photParams=params)

                    detectors.append(detector)

            if not hasattr(self, 'bandpassDict'):
                if self.noise_and_background is not None:
                    if self.obs_metadata.m5 is None:
                        raise RuntimeError('WARNING  in GalSimCatalog; you did not specify m5 in your '+
                                            'obs_metadata. m5 is required in order to add noise to your images')

                    for name in self.bandpassNames:
                        if name not in self.obs_metadata.m5:
                            raise RuntimeError('WARNING in GalSimCatalog; your obs_metadata does not have ' +
                                                 'm5 values for all of your bandpasses \n' +
                                                 'bandpass has: %s \n' % self.bandpassNames.__repr__() +
                                                 'm5 has: %s ' % self.obs_metadata.m5.keys().__repr__())

                    if self.obs_metadata.seeing is None:
                        raise RuntimeError('WARNING  in GalSimCatalog; you did not specify seeing in your '+
                                            'obs_metadata.  seeing is required in order to add noise to your images')

                    for name in self.bandpassNames:
                        if name not in self.obs_metadata.seeing:
                            raise RuntimeError('WARNING in GalSimCatalog; your obs_metadata does not have ' +
                                                 'seeing values for all of your bandpasses \n' +
                                                 'bandpass has: %s \n' % self.bandpassNames.__repr__() +
                                                 'seeing has: %s ' % self.obs_metadata.seeing.keys().__repr__())

                self.bandpassDict, hardwareDict = BandpassDict.loadBandpassesFromFiles(bandpassNames=self.bandpassNames,
                                             filedir=self.bandpassDir,
                                             bandpassRoot=self.bandpassRoot,
                                             componentList=self.componentList,
                                             atmoTransmission=os.path.join(self.bandpassDir, self.atmoTransmissionName))

            self.galSimInterpreter = GalSimInterpreter(obs_metadata=self.obs_metadata, epoch=self.db_obj.epoch, detectors=detectors,
                                                       bandpassDict=self.bandpassDict, noiseWrapper=self.noise_and_background,
                                                       seed=self.seed)

            self.galSimInterpreter.setPSF(PSF=self.PSF)


    def write_images(self, nameRoot=None):
        """
        Writes the FITS images associated with this InstanceCatalog.

        Cannot be called before write_catalog is called.

        @param [in] nameRoot is an optional string prepended to the names
        of the FITS images.  The FITS images will be named

        @param [out] namesWritten is a list of the names of the FITS files generated

        nameRoot_DetectorName_FilterName.fits

        (e.g. myImages_R_0_0_S_1_1_y.fits for an LSST-like camera with
        nameRoot = 'myImages')
        """
        namesWritten = self.galSimInterpreter.writeImages(nameRoot=nameRoot)

        return namesWritten

class GalSimGalaxies(GalSimBase, AstrometryGalaxies, EBVmixin):
    """
    This is a GalSimCatalog class for galaxy components (i.e. objects that are shaped
    like Sersic profiles).

    See the docstring in GalSimBase for explanation of how this class should be used.
    """

    catalog_type = 'galsim_galaxy'
    galsim_type = 'sersic'
    default_columns = [('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'sersic', (str,6))]

class GalSimAgn(GalSimBase, AstrometryGalaxies, EBVmixin):
    """
    This is a GalSimCatalog class for AGN.

    See the docstring in GalSimBase for explanation of how this class should be used.
    """
    catalog_type = 'galsim_agn'
    galsim_type = 'pointSource'
    default_columns = [('galacticAv', 0.1, float),
                      ('galacticRv', 3.1, float),
                      ('galSimType', 'pointSource', (str,11)),
                      ('majorAxis', 0.0, float),
                      ('minorAxis', 0.0, float),
                      ('sindex', 0.0, float),
                      ('positionAngle', 0.0, float),
                      ('halfLightRadius', 0.0, float),
                      ('internalAv', 0.0, float),
                      ('internalRv', 0.0, float)]

class GalSimStars(GalSimBase, AstrometryStars, EBVmixin):
    """
    This is a GalSimCatalog class for stars.

    See the docstring in GalSimBase for explanation of how this class should be used.
    """
    catalog_type = 'galsim_stars'
    galsim_type = 'pointSource'
    default_columns = [('galacticAv', 0.1, float),
                      ('galacticRv', 3.1, float),
                      ('galSimType', 'pointSource', (str,11)),
                      ('internalAv', 0.0, float),
                      ('internalRv', 0.0, float),
                      ('redshift', 0.0, float),
                      ('majorAxis', 0.0, float),
                      ('minorAxis', 0.0, float),
                      ('sindex', 0.0, float),
                      ('positionAngle', 0.0, float),
                      ('halfLightRadius', 0.0, float)]
