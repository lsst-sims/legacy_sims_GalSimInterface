from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from lsst.sims.coordUtils import raDecFromPixelCoordinates

__all__ = ["get_center_of_detector"]


def get_center_of_detector(detector, camera, obs, epoch=2000.0):

    pixelSystem = detector.makeCameraSys(PIXELS)
    centerPoint = detector.getCenter(FOCAL_PLANE)
    centerPixel = camera.transform(centerPoint, pixelSystem).getPoint()
    xPix = centerPixel.getX()
    yPix = centerPixel.getY()
    ra, dec = raDecFromPixelCoordinates([xPix], [yPix], [detector.getName()],
                                        camera=camera, obs_metadata=obs, epoch=epoch)

    return ra[0], dec[0]

