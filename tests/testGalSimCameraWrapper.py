import unittest
import numpy as np
import lsst.utils.tests

from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import raDecFromAltAz
from lsst.sims.coordUtils import pixelCoordsFromRaDec
from lsst.sims.coordUtils import _pixelCoordsFromRaDec
from lsst.sims.coordUtils import raDecFromPixelCoords
from lsst.sims.coordUtils import _raDecFromPixelCoords
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords
from lsst.sims.coordUtils import pixelCoordsFromPupilCoords

from lsst.sims.GalSimInterface import GalSimCameraWrapper
from lsst.sims.GalSimInterface import LSSTCameraWrapper

import lsst.afw.cameraGeom.testUtils as camTestUtils
from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.afw.cameraGeom import TAN_PIXELS, FIELD_ANGLE, PIXELS

from lsst.sims.coordUtils import pupilCoordsFromPixelCoordsLSST
from lsst.sims.coordUtils import pixelCoordsFromPupilCoordsLSST
from lsst.sims.coordUtils import raDecFromPixelCoordsLSST
from lsst.sims.coordUtils import lsst_camera

from lsst.sims.coordUtils import clean_up_lsst_camera

def setup_module(module):
    lsst.utils.tests.init()


class Camera_Wrapper_Test_Class(unittest.TestCase):

    longMessage = True

    @classmethod
    def tearDownClass(cls):
        clean_up_lsst_camera()

    def test_generic_camera_wrapper(self):
        """
        Test that GalSimCameraWrapper wraps its methods as expected.
        This is mostly to catch changes in afw API.
        """
        camera = camTestUtils.CameraWrapper().camera
        camera_wrapper = GalSimCameraWrapper(camera)

        obs_mjd = ObservationMetaData(mjd=60000.0)
        ra, dec = raDecFromAltAz(35.0, 112.0, obs_mjd)
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=obs_mjd.mjd,
                                  rotSkyPos=22.4)

        rng = np.random.RandomState(8124)

        for detector in camera:
            name = detector.getName()
            bbox = camera[name].getBBox()
            bbox_wrapper = camera_wrapper.getBBox(name)
            self.assertEqual(bbox.getMinX(), bbox_wrapper.getMinX())
            self.assertEqual(bbox.getMaxX(), bbox_wrapper.getMaxX())
            self.assertEqual(bbox.getMinY(), bbox_wrapper.getMinY())
            self.assertEqual(bbox.getMaxY(), bbox_wrapper.getMaxY())

            center_point = camera[name].getCenter(FOCAL_PLANE)
            pixel_system = camera[name].makeCameraSys(PIXELS)
            center_pix = camera.transform(center_point, FOCAL_PLANE, pixel_system)
            center_pix_wrapper = camera_wrapper.getCenterPixel(name)
            self.assertEqual(center_pix.getX(), center_pix_wrapper.getX())
            self.assertEqual(center_pix.getY(), center_pix_wrapper.getY())

            pupil_system = camera[name].makeCameraSys(FIELD_ANGLE)
            center_pupil = camera.transform(center_point, FOCAL_PLANE, pupil_system)
            center_pupil_wrapper = camera_wrapper.getCenterPupil(name)
            self.assertEqual(center_pupil.getX(), center_pupil_wrapper.getX())
            self.assertEqual(center_pupil.getY(), center_pupil_wrapper.getY())

            corner_pupil_wrapper = camera_wrapper.getCornerPupilList(name)
            corner_point_list = camera[name].getCorners(FOCAL_PLANE)
            for point in corner_point_list:
                point_pupil = camera.transform(point, FOCAL_PLANE, pupil_system)
                dd_min = 1.0e10
                for wrapper_point in corner_pupil_wrapper:
                    dd = np.sqrt((point_pupil.getX()-wrapper_point.getX())**2 +
                                 (point_pupil.getY()-wrapper_point.getY())**2)

                    if dd < dd_min:
                        dd_min = dd
                self.assertLess(dd_min, 1.0e-20)

            xpix_min = None
            xpix_max = None
            ypix_min = None
            ypix_max = None
            focal_to_tan_pix = camera[name].getTransform(FOCAL_PLANE, TAN_PIXELS)
            for point in corner_point_list:
                pixel_point = focal_to_tan_pix.applyForward(point)
                xx = pixel_point.getX()
                yy = pixel_point.getY()
                if xpix_min is None or xx<xpix_min:
                    xpix_min = xx
                if ypix_min is None or yy<ypix_min:
                    ypix_min = yy
                if xpix_max is None or xx>xpix_max:
                    xpix_max = xx
                if ypix_max is None or yy>ypix_max:
                    ypix_max = yy

            pix_bounds_wrapper = camera_wrapper.getTanPixelBounds(name)
            self.assertEqual(pix_bounds_wrapper[0], xpix_min)
            self.assertEqual(pix_bounds_wrapper[1], xpix_max)
            self.assertEqual(pix_bounds_wrapper[2], ypix_min)
            self.assertEqual(pix_bounds_wrapper[3], ypix_max)

            x_pup = rng.random_sample(10)*0.005-0.01
            y_pup = rng.random_sample(10)*0.005-0.01
            x_pix, y_pix = pixelCoordsFromPupilCoords(x_pup, y_pup, chipName=name,
                                                      camera=camera)

            (x_pix_wrapper,
             y_pix_wrapper) = camera_wrapper.pixelCoordsFromPupilCoords(x_pup, y_pup,
                                                                        name, obs)

            nan_x = np.where(np.isnan(x_pix))
            self.assertEqual(len(nan_x[0]), 0)
            np.testing.assert_array_equal(x_pix, x_pix_wrapper)
            np.testing.assert_array_equal(y_pix, y_pix_wrapper)

            x_pix = rng.random_sample(10)*100.0-200.0
            y_pix = rng.random_sample(10)*100.0-200.0
            x_pup, y_pup = pupilCoordsFromPixelCoords(x_pix, y_pix, chipName=name,
                                                      camera=camera)

            (x_pup_wrapper,
             y_pup_wrapper) = camera_wrapper.pupilCoordsFromPixelCoords(x_pix, y_pix, name,
                                                                        obs)

            nan_x = np.where(np.isnan(x_pup))
            self.assertEqual(len(nan_x[0]), 0)
            np.testing.assert_array_equal(x_pup, x_pup_wrapper)
            np.testing.assert_array_equal(y_pup, y_pup_wrapper)

            ra, dec = raDecFromPixelCoords(x_pix, y_pix, name, camera=camera,
                                           obs_metadata=obs)

            (ra_wrapper,
             dec_wrapper) = camera_wrapper.raDecFromPixelCoords(x_pix, y_pix, name, obs)

            nan_ra = np.where(np.isnan(ra))
            self.assertEqual(len(nan_ra[0]), 0)
            np.testing.assert_array_equal(ra, ra_wrapper)
            np.testing.assert_array_equal(dec, dec_wrapper)

            ra, dec = _raDecFromPixelCoords(x_pix, y_pix, name, camera=camera,
                                            obs_metadata=obs)

            (ra_wrapper,
             dec_wrapper) = camera_wrapper._raDecFromPixelCoords(x_pix, y_pix, name, obs)

            nan_ra = np.where(np.isnan(ra))
            self.assertEqual(len(nan_ra[0]), 0)
            np.testing.assert_array_equal(ra, ra_wrapper)
            np.testing.assert_array_equal(dec, dec_wrapper)

            ra = obs.pointingRA + (rng.random_sample(10)*50.0-100.0)/60.0
            dec = obs.pointingDec + (rng.random_sample(10)*50.0-100.0)/60.0

            x_pix, y_pix = pixelCoordsFromRaDec(ra, dec, chipName=name, camera=camera,
                                                obs_metadata=obs)

            (x_pix_wrapper,
             y_pix_wrapper) = camera_wrapper.pixelCoordsFromRaDec(ra, dec, chipName=name,
                                                                  obs_metadata=obs)

            nan_x = np.where(np.isnan(x_pix))
            self.assertEqual(len(nan_x[0]), 0)
            np.testing.assert_array_equal(x_pix, x_pix_wrapper)
            np.testing.assert_array_equal(y_pix, y_pix_wrapper)

            ra = np.radians(ra)
            dec = np.radians(dec)

            x_pix, y_pix = _pixelCoordsFromRaDec(ra, dec, chipName=name, camera=camera,
                                                 obs_metadata=obs)

            (x_pix_wrapper,
             y_pix_wrapper) = camera_wrapper._pixelCoordsFromRaDec(ra, dec, chipName=name,
                                                                   obs_metadata=obs)

            nan_x = np.where(np.isnan(x_pix))
            self.assertEqual(len(nan_x[0]), 0)
            np.testing.assert_array_equal(x_pix, x_pix_wrapper)
            np.testing.assert_array_equal(y_pix, y_pix_wrapper)

        del camera

    def test_LSST_camera_wrapper(self):
        """
        Test that LSSTCameraWrapper wraps its methods as expected.

        Recall that the LSSTCameraWrapper applies the 90 degree rotation
        to go from DM pixel coordinates to Camera team pixel coordinates.
        Namely,

        Camera +y = DM +x
        Camera +x = DM -y
        """
        camera = lsst_camera()
        camera_wrapper = LSSTCameraWrapper()

        obs_mjd = ObservationMetaData(mjd=60000.0)
        ra, dec = raDecFromAltAz(35.0, 112.0, obs_mjd)
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=obs_mjd.mjd,
                                  rotSkyPos=22.4,
                                  bandpassName='u')

        rng = np.random.RandomState(8124)

        for detector in camera:
            name = detector.getName()
            bbox = camera[name].getBBox()
            bbox_wrapper = camera_wrapper.getBBox(name)
            self.assertEqual(bbox.getMinX(), bbox_wrapper.getMinY())
            self.assertEqual(bbox.getMaxX(), bbox_wrapper.getMaxY())
            self.assertEqual(bbox.getMinY(), bbox_wrapper.getMinX())
            self.assertEqual(bbox.getMaxY(), bbox_wrapper.getMaxX())
            self.assertGreater(bbox_wrapper.getMaxY()-bbox_wrapper.getMinY(),
                               bbox_wrapper.getMaxX()-bbox_wrapper.getMinX())

            center_point = camera[name].getCenter(FOCAL_PLANE)
            pixel_system = camera[name].makeCameraSys(PIXELS)
            center_pix = camera.transform(center_point, FOCAL_PLANE, pixel_system)
            center_pix_wrapper = camera_wrapper.getCenterPixel(name)
            self.assertEqual(center_pix.getX(), center_pix_wrapper.getY())
            self.assertEqual(center_pix.getY(), center_pix_wrapper.getX())

            # Note that DM and the Camera team agree on the orientation
            # of the pupil coordinate/field angle axes
            pupil_system = camera[name].makeCameraSys(FIELD_ANGLE)
            center_pupil = camera.transform(center_point, FOCAL_PLANE, pupil_system)
            center_pupil_wrapper = camera_wrapper.getCenterPupil(name)
            self.assertEqual(center_pupil.getX(), center_pupil_wrapper.getX())
            self.assertEqual(center_pupil.getY(), center_pupil_wrapper.getY())

            corner_pupil_wrapper = camera_wrapper.getCornerPupilList(name)
            corner_point_list = camera[name].getCorners(FOCAL_PLANE)
            for point in corner_point_list:
                point_pupil = camera.transform(point, FOCAL_PLANE, pupil_system)
                dd_min = 1.0e10
                for wrapper_point in corner_pupil_wrapper:
                    dd = np.sqrt((point_pupil.getX()-wrapper_point.getX())**2 +
                                 (point_pupil.getY()-wrapper_point.getY())**2)

                    if dd < dd_min:
                        dd_min = dd
                self.assertLess(dd_min, 1.0e-20)

            xpix_min = None
            xpix_max = None
            ypix_min = None
            ypix_max = None
            focal_to_tan_pix = camera[name].getTransform(FOCAL_PLANE, TAN_PIXELS)
            for point in corner_point_list:
                pixel_point = focal_to_tan_pix.applyForward(point)
                xx = pixel_point.getX()
                yy = pixel_point.getY()
                if xpix_min is None or xx<xpix_min:
                    xpix_min = xx
                if ypix_min is None or yy<ypix_min:
                    ypix_min = yy
                if xpix_max is None or xx>xpix_max:
                    xpix_max = xx
                if ypix_max is None or yy>ypix_max:
                    ypix_max = yy

            pix_bounds_wrapper = camera_wrapper.getTanPixelBounds(name)
            self.assertEqual(pix_bounds_wrapper[0], ypix_min)
            self.assertEqual(pix_bounds_wrapper[1], ypix_max)
            self.assertEqual(pix_bounds_wrapper[2], xpix_min)
            self.assertEqual(pix_bounds_wrapper[3], xpix_max)

            # generate some random pupil coordinates;
            # verify that the relationship between the DM and Camera team
            # pixel coordinates corresponding to those pupil coordinates
            # is as expected
            x_pup = rng.random_sample(10)*0.005-0.01
            y_pup = rng.random_sample(10)*0.005-0.01
            x_pix, y_pix = pixelCoordsFromPupilCoordsLSST(x_pup, y_pup, chipName=name,
                                                          band=obs.bandpass)

            (x_pix_wrapper,
             y_pix_wrapper) = camera_wrapper.pixelCoordsFromPupilCoords(x_pup, y_pup,
                                                                        name, obs)

            nan_x = np.where(np.isnan(x_pix))
            self.assertEqual(len(nan_x[0]), 0)
            np.testing.assert_allclose(x_pix-center_pix.getX(),
                                       y_pix_wrapper-center_pix_wrapper.getY(),
                                       atol=1.0e-10, rtol=0.0)
            np.testing.assert_allclose(y_pix-center_pix.getY(),
                                       center_pix_wrapper.getX()-x_pix_wrapper,
                                       atol=1.0e-10, rtol=0.0)

            # use camera_wrapper.pupilCoordsFromPixelCoords to go back to pupil
            # coordinates from x_pix_wrapper, y_pix_wrapper; make sure you get
            # the original pupil coordinates back out
            (x_pup_wrapper,
             y_pup_wrapper) = camera_wrapper.pupilCoordsFromPixelCoords(x_pix_wrapper,
                                                                        y_pix_wrapper,
                                                                        name, obs)
            msg = 'worst error %e' % np.abs(x_pup-x_pup_wrapper).max()
            np.testing.assert_allclose(x_pup, x_pup_wrapper, atol=1.0e-10, rtol=0.0, err_msg=msg)
            msg = 'worst error %e' % np.abs(y_pup-y_pup_wrapper).max()
            np.testing.assert_allclose(y_pup, y_pup_wrapper, atol=1.0e-10, rtol=0.0, err_msg=msg)

            # generate some random sky coordinates; verify that the methods that
            # convert between (RA, Dec) and pixel coordinates behave as expected.
            # NOTE: x_pix, y_pix will be in DM pixel coordinate convention
            x_pix = bbox.getMinX() + rng.random_sample(10)*(bbox.getMaxX()-bbox.getMinX())
            y_pix = bbox.getMinY() + rng.random_sample(10)*(bbox.getMaxY()-bbox.getMinY())

            ra, dec = raDecFromPixelCoordsLSST(x_pix, y_pix, name, obs_metadata=obs,
                                               band=obs.bandpass)

            (ra_wrapper,
             dec_wrapper) = camera_wrapper.raDecFromPixelCoords(2.0*center_pix.getY()-y_pix,
                                                                x_pix, name, obs)

            nan_ra = np.where(np.isnan(ra))
            self.assertEqual(len(nan_ra[0]), 0)
            np.testing.assert_allclose(ra, ra_wrapper, atol=1.0e-10, rtol=0.0)
            np.testing.assert_allclose(dec, dec_wrapper, atol=1.0e-10, rtol=0.0)

            # make sure that the method that returns RA, Dec in radians agrees with
            # the method that returns RA, Dec in degrees
            (ra_rad,
             dec_rad) = camera_wrapper._raDecFromPixelCoords(2.0*center_pix.getY()-y_pix,
                                                             x_pix, name, obs)

            np.testing.assert_allclose(np.radians(ra_wrapper), ra_rad, atol=1.0e-10, rtol=0.0)
            np.testing.assert_allclose(np.radians(dec_wrapper), dec_rad, atol=1.0e-10, rtol=0.0)

            # Go back to pixel coordinates with pixelCoordsFromRaDec; verify that
            # the result relates to the original DM pixel coordinates as expected
            # (x_pix_inv, y_pix_inv will be in Camera pixel coordinates)
            (x_pix_inv,
             y_pix_inv) = camera_wrapper.pixelCoordsFromRaDec(ra_wrapper, dec_wrapper,
                                                              chipName=name,
                                                              obs_metadata=obs)

            np.testing.assert_allclose(y_pix_inv, x_pix, atol=1.0e-4, rtol=0.0)
            np.testing.assert_allclose(x_pix_inv, 2.0*center_pix.getY()-y_pix, atol=1.0e-4, rtol=0.0)

            ra = np.radians(ra_wrapper)
            dec = np.radians(dec_wrapper)

            # check that the the method that accepts RA, Dec in radians agrees with the
            # method that accepts RA, Dec in degrees
            (x_pix_wrapper,
             y_pix_wrapper) = camera_wrapper._pixelCoordsFromRaDec(ra, dec, chipName=name,
                                                                   obs_metadata=obs)

            np.testing.assert_allclose(x_pix_inv, x_pix_wrapper, atol=1.0e-10, rtol=0.0)
            np.testing.assert_allclose(y_pix_inv, y_pix_wrapper, atol=1.0e-10, rtol=0.0)

        del camera
        del camera_wrapper
        del lsst_camera._lsst_camera

    def test_dmPixFromCameraPix(self):
        """
        Test that the method to return DM pixel coordinates from
        Camera Team pixel coordinates works.
        """
        camera = lsst_camera()
        camera_wrapper = LSSTCameraWrapper()
        obs = ObservationMetaData(bandpassName='u')

        npts = 100
        rng = np.random.RandomState(1824)
        dm_x_pix_list = rng.random_sample(npts)*4000.0
        dm_y_pix_list = rng.random_sample(npts)*4000.0
        name_list = []
        for det in camera:
            name_list.append(det.getName())
        chip_name_list = rng.choice(name_list, size=npts)

        (xPup_list,
         yPup_list) = pupilCoordsFromPixelCoordsLSST(dm_x_pix_list,
                                                     dm_y_pix_list,
                                                     chipName=chip_name_list,
                                                     band=obs.bandpass)

        (cam_x_pix_list,
         cam_y_pix_list) = camera_wrapper.pixelCoordsFromPupilCoords(xPup_list,
                                                                     yPup_list,
                                                                     chip_name_list,
                                                                     obs)

        (dm_x_test,
         dm_y_test) = camera_wrapper.dmPixFromCameraPix(cam_x_pix_list,
                                                        cam_y_pix_list,
                                                        chip_name_list)

        np.testing.assert_array_almost_equal(dm_x_test, dm_x_pix_list,
                                             decimal=4)
        np.testing.assert_array_almost_equal(dm_y_test, dm_y_pix_list,
                                             decimal=4)

        # test transformations made one at a time
        for ii in range(len(cam_x_pix_list)):
            dm_x, dm_y = camera_wrapper.dmPixFromCameraPix(cam_x_pix_list[ii],
                                                           cam_y_pix_list[ii],
                                                           chip_name_list[ii])

            self.assertAlmostEqual(dm_x_pix_list[ii], dm_x, 4)
            self.assertAlmostEqual(dm_y_pix_list[ii], dm_y, 4)

        # test case where an array of points is on a single chip
        chip_name = chip_name_list[10]

        (xPup_list,
         yPup_list) = pupilCoordsFromPixelCoordsLSST(dm_x_pix_list,
                                                     dm_y_pix_list,
                                                     chipName=chip_name,
                                                     band=obs.bandpass)

        (cam_x_pix_list,
         cam_y_pix_list) = camera_wrapper.pixelCoordsFromPupilCoords(xPup_list,
                                                                     yPup_list,
                                                                     chip_name,
                                                                     obs)

        (dm_x_test,
         dm_y_test) = camera_wrapper.dmPixFromCameraPix(cam_x_pix_list,
                                                        cam_y_pix_list,
                                                        chip_name)

        np.testing.assert_array_almost_equal(dm_x_test, dm_x_pix_list,
                                             decimal=4)
        np.testing.assert_array_almost_equal(dm_y_test, dm_y_pix_list,
                                             decimal=4)

        del camera
        del camera_wrapper
        del lsst_camera._lsst_camera

    def test_camPixFromDMpix(self):
        """
        test that camPixFromDMpix inverts dmPixFromCamPix
        """
        camera_wrapper = LSSTCameraWrapper()
        rng = np.random.RandomState()
        npts = 200
        cam_x_in = rng.random_sample(npts)*4000.0
        cam_y_in = rng.random_sample(npts)*4000.0
        dm_x, dm_y = camera_wrapper.dmPixFromCameraPix(cam_x_in, cam_y_in, 'R:1,1 S:2,2')
        cam_x, cam_y = camera_wrapper.cameraPixFromDMPix(dm_x, dm_y, 'R:1,1 S:2,2')
        np.testing.assert_array_almost_equal(cam_x_in, cam_x, decimal=10)
        np.testing.assert_array_almost_equal(cam_y_in, cam_y, decimal=10)
        del camera_wrapper
        del lsst_camera._lsst_camera


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
