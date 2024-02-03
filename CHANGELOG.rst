Changes from v0.5 to v0.6
=========================


API Changes
-----------


New Features
------------
- Added zernikeXYAberration to map pupil coordinates to focal plane coordinates.
- Added hexapolar util to compute hexapolar grids.
- Added jmin to zernikePyramid to enable j<4 plots.
- Added imin argument to Asphere.
- Add SDSS telescope description.


Performance Improvements
------------------------


Bug Fixes
---------
- Fixed bug where withLocallyShiftedOptic used the top-level rotation instead of the subitem rotation.
