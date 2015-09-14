"""
This script illustrates how to add noise to a series of FITS images using stars
"""

import os
import galsim
from lsst.sims.catalogs.generation.db import CatalogDBObject
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catUtils.baseCatalogModels import StarObj, OpSim3_61DBObject
from lsst.sims.GalSimInterface import GalSimStars, SNRdocumentPSF, ExampleCCDNoise
from lsst.sims.photUtils import LSSTdefaults

#if you want to use the actual LSST camera
#from lsst.obs.lsstSim import LsstSimMapper

class testGalSimStarsNoiseless(GalSimStars):
    #only draw images for u and g bands (for speed)
    bandpassNames = ['u','g']

    #defined in galSimInterface/galSimUtilities.py
    PSF = SNRdocumentPSF()

    #If you want to use the LSST camera, uncomment the line below.
    #You can similarly assign any camera object you want here
    #camera = LsstSimMapper().camera



class testGalSimStarsWithNoise(testGalSimStarsNoiseless):

    #defined in galSimInterface/galSimUtilities.py
    noise_and_background = ExampleCCDNoise(seed=99)

#select an OpSim pointing
obsMD = OpSim3_61DBObject()
raw_obs_metadata = obsMD.getObservationMetaData(88625744, 0.1, makeCircBounds = True)


defaults = LSSTdefaults()
obs_metadata = ObservationMetaData(unrefractedRA=raw_obs_metadata.unrefractedRA,
                                   unrefractedDec=raw_obs_metadata.unrefractedDec,
                                   boundType='circle',
                                   boundLength=0.1,
                                   mjd=raw_obs_metadata.mjd,
                                   rotSkyPos=raw_obs_metadata.rotSkyPos,
                                   bandpassName=['u','g'],
                                   m5=[defaults.m5('u'), defaults.m5('g')],
                                   seeing=[defaults.seeing('u'), defaults.seeing('g')])


#grab a database of galaxies (in this case, galaxy bulges)
stars = CatalogDBObject.from_objid('allstars')

#now append a bunch of objects with 2D sersic profiles to our output file
stars_noiseless = testGalSimStarsNoiseless(stars, obs_metadata=obs_metadata)

stars_noiseless.write_catalog('galSim_NoiselessStars_example.txt', chunk_size=10000)
stars_noiseless.write_images(nameRoot='noiselessStars')

stars_noisy = testGalSimStarsWithNoise(stars, obs_metadata=obs_metadata)
stars_noisy.write_catalog('galSim_NoisyStars_example.txt', chunk_size=10000)
stars_noisy.write_images(nameRoot='noisyStars')
