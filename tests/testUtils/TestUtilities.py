import os
import numpy
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from lsst.sims.coordUtils import raDecFromPixelCoordinates
from lsst.sims.utils import radiansFromArcsec

__all__ = ["create_text_catalog"]


def create_text_catalog(obs, file_name, xDisplacement, yDisplacement, hlr=None, mag_norm=None):

    if os.path.exists(file_name):
        os.unlink(file_name)

    dxList = radiansFromArcsec(xDisplacement)
    dyList = radiansFromArcsec(yDisplacement)

    if hlr is None:
        hlr = [2.0]*len(dxList)

    if mag_norm is None:
        mag_norm = [21.0]*len(dxList)


    with open(file_name,'w') as outFile:
        outFile.write('# test_id ra dec hlr mag_norm\n')
        for ix, (dx, dy, halfLight, magNorm) in enumerate(zip(dxList, dyList, hlr, mag_norm)):

            rr = numpy.degrees(obs._unrefractedRA+dx)
            dd = numpy.degrees(obs._unrefractedDec+dy)

            outFile.write('%d %.9f %.9f %.9f %.9f\n' % (ix, rr, dd, halfLight, magNorm))
