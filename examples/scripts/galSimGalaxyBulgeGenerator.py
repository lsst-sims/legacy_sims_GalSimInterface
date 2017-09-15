"""
This script shows how to use our GalSim interface to create FITS images of
galaxy bulges (or any sersic profile)
"""

import os
from lsst.utils import getPackageDir
from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.catUtils.baseCatalogModels import GalaxyBulgeObj
from lsst.sims.GalSimInterface import GalSimGalaxies, SNRdocumentPSF
from lsst.sims.GalSimInterface import LSSTCameraWrapper

#if you want to use the actual LSST camera
#from lsst.obs.lsstSim import LsstSimMapper

class testGalSimGalaxies(GalSimGalaxies):
    #only draw images for u and g bands (for speed)
    bandpassNames = ['u','g']

    PSF = SNRdocumentPSF()

#select an OpSim pointing
opsimdb = os.path.join(getPackageDir('sims_data'), 'OpSimData',
                       'opsimblitz1_1133_sqlite.db')
obs_gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')
obs_list = obs_gen.getObservationMetaData(obsHistID=10, boundLength=0.05)
obs_metadata = obs_list[0]

#grab a database of galaxies (in this case, galaxy bulges)
gals = CatalogDBObject.from_objid('galaxyBulge')

#now append a bunch of objects with 2D sersic profiles to our output file
galaxy_galSim = testGalSimGalaxies(gals, obs_metadata=obs_metadata)
galaxy_galSim.camera_wrapper = LSSTCameraWrapper()

galaxy_galSim.write_catalog('galSim_bulge_example.txt', chunk_size=10000)
galaxy_galSim.write_images(nameRoot='bulge')
