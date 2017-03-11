"""
This script shows how incorporate noise in images of galaxies
"""

import os
from lsst.utils import getPackageDir
from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catUtils.baseCatalogModels import GalaxyBulgeObj
from lsst.sims.GalSimInterface import GalSimGalaxies, ExampleCCDNoise, \
                                               SNRdocumentPSF

from lsst.sims.photUtils import LSSTdefaults

#if you want to use the actual LSST camera
#from lsst.obs.lsstSim import LsstSimMapper

class testGalSimGalaxiesNoiseless(GalSimGalaxies):
    #only draw images for u and g bands (for speed)
    bandpassNames = ['u','g']

    #If you want to use the LSST camera, uncomment the line below.
    #You can similarly assign any camera object you want here
    #camera = LsstSimMapper().camera

    PSF = SNRdocumentPSF()

class testGalSimGalaxiesNoisy(testGalSimGalaxiesNoiseless):

    #defined in galSimInterface/galSimUtilities.py
    noise_and_background = ExampleCCDNoise(99)

#select an OpSim pointing
opsimdb = os.path.join(getPackageDir('sims_data'), 'OpSimData',
                       'opsimblitz1_1133_sqlite.db')
obs_gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')
obs_list = obs_gen.getObservationMetaData(obsHistID=10, boundLength=0.05)
raw_obs_metadata = obs_list[0]

defaults = LSSTdefaults()
obs_metadata = ObservationMetaData(pointingRA=raw_obs_metadata.pointingRA,
                                   pointingDec=raw_obs_metadata.pointingDec,
                                   boundType='circle',
                                   boundLength=0.05,
                                   mjd=raw_obs_metadata.mjd,
                                   rotSkyPos=raw_obs_metadata.rotSkyPos,
                                   bandpassName=['u','g'],
                                   m5=[defaults.m5('u'), defaults.m5('g')],
                                   seeing=[defaults.FWHMeff('u'), defaults.FWHMeff('g')])

#grab a database of galaxies (in this case, galaxy bulges)
gals = CatalogDBObject.from_objid('galaxyBulge')

#now append a bunch of objects with 2D sersic profiles to our output file
gal_noiseless = testGalSimGalaxiesNoiseless(gals, obs_metadata=obs_metadata)

gal_noiseless.write_catalog('galSim_NoiselessGalaxies_example.txt', chunk_size=10000)
gal_noiseless.write_images(nameRoot='noiselessGalaxies')

gal_noisy = testGalSimGalaxiesNoisy(gals, obs_metadata=obs_metadata)
gal_noisy.write_catalog('galSim_NoisyGalaxies_example.txt', chunk_size=10000)
gal_noisy.write_images(nameRoot='noisyGalaxies')
