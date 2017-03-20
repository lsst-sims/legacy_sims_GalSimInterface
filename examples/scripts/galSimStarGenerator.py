"""
This script shows how to use our GalSim interface to generate FITS images of stars
"""

import os
from lsst.utils import getPackageDir
from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.catUtils.baseCatalogModels import StarObj
from lsst.sims.GalSimInterface import SNRdocumentPSF, GalSimStars

#if you want to use the actual LSST camera
#from lsst.obs.lsstSim import LsstSimMapper

class testGalSimStars(GalSimStars):
    #only draw images for u and g bands (for speed)
    bandpassNames = ['u','g']

    #defined in galSimInterface/galSimUtilities.py
    PSF = SNRdocumentPSF()

    #If you want to use the LSST camera, uncomment the line below.
    #You can similarly assign any camera object you want here
    #camera = LsstSimMapper().camera



#select an OpSim pointing
#select an OpSim pointing
opsimdb = os.path.join(getPackageDir('sims_data'), 'OpSimData',
                       'opsimblitz1_1133_sqlite.db')
obs_gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')
obs_list = obs_gen.getObservationMetaData(obsHistID=10, boundLength=0.05)
obs_metadata = obs_list[0]

#grab a database of stars
stars = CatalogDBObject.from_objid('allstars')

#now append a bunch of objects with 2D sersic profiles to our output file
stars_galSim = testGalSimStars(stars, obs_metadata=obs_metadata)

stars_galSim.write_catalog('galSim_star_example.txt', chunk_size=100)
stars_galSim.write_images(nameRoot='star')
