from builtins import zip
import os
import numpy
from lsst.sims.utils import radiansFromArcsec, icrsFromObserved

__all__ = ["create_text_catalog"]


def create_text_catalog(obs, file_name, raDisplacement, decDisplacement, \
                        hlr=None, mag_norm=None, pa=None):
    """
    Create a text file containing objects that can be read in by a fileDBObject class.

    @param [in] obs is an ObservationMetaData specifying the pointing on which the catalog
    will be centered

    @param [in] file_name is the name of the file to be created.  If a file already exists with that
    name, throw an error.

    @param [in] raDisplacement is a numpy array listing the RA displacements of objects from the
    pointing's center in arcseconds

    @param [in] decDisplacement is a numpy array listing the Dec displacements of objects from the
    pointings' center in arcseconds

    @param [in] hlr is an optional list of half light radii in arcseconds

    @param [in] mag_norm is an optional list of the objects' magnitude normalizations

    @param [in] pa is an optional list of the objects' position angles in degrees
    """

    if os.path.exists(file_name):
        os.unlink(file_name)

    raDisplacementList = radiansFromArcsec(raDisplacement)
    decDisplacementList = radiansFromArcsec(decDisplacement)

    if hlr is None:
        hlr = [2.0]*len(raDisplacement)

    if mag_norm is None:
        mag_norm = [20.0]*len(raDisplacement)

    if pa is None:
        pa = [0.0]*len(raDisplacement)


    with open(file_name,'w') as outFile:
        outFile.write('# test_id ra dec hlr mag_norm pa\n')

        for ix, (dx, dy, halfLight, magNorm, pp) in \
        enumerate(zip(raDisplacementList, decDisplacementList, hlr, mag_norm, pa)):

            rr = numpy.degrees(obs._pointingRA+dx)
            dd = numpy.degrees(obs._pointingDec+dy)

            outFile.write('%d %.9f %.9f %.9f %.9f %.9f\n' % (ix, rr, dd, halfLight, magNorm, pp))
