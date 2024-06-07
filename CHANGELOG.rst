Changes from v0.6 to v0.7
=========================


API Changes
-----------


New Features
------------
- Added intersect/reflect/refract methods directly to `Optic.Interface`
  objects.
- Added x_origin and y_origin arguments to batoid.Zernike.
- Add Rubin as-built v3.14 telescope description yamls.


Performance Improvements
------------------------


Bug Fixes
---------
- Fixed error in ComCam definition files where the wrong filter
  thickness was used for the i, z, and y filters.
- Photons experiencing total internal reflection are now marked
  as failed.
- Catch rank deficient matrix in batoid.zernike.
- Force RayVector constructor to copy input arrays.
- Fixed bug in ComCam yaml files where the wrong filter thickness
  was used for the i, z, and y filters.
