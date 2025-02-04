Changes from v0.7 to v0.8
=========================


API Changes
-----------


New Features
------------
- Ability to specify 'local' as coordSys of rotation center in
  global rotation methods.
- Added `withGloballyRotatedOptic` method.
- Added coordSys option for draw3d.
- Added a UPS table
- Added ComCam designs with the spider.
- Added include_vignetted option to various Zernike functions.


Performance Improvements
------------------------
- Use galsim.zernike.zernikeGradBases in zernikeTA


Bug Fixes
---------
