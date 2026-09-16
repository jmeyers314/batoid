Changes from v0.8 to v0.9
=========================


API Changes
-----------


New Features
------------
- Implemented finite-distance point sources in the `RayVector` factories
  `asPolar`, `asSpokes`, `asGrid`, and `fromStop`.  The `source` keyword was
  previously accepted but unimplemented; rays are now launched from `source`
  and advanced to `backDist` ahead of the stop.
- Added `conjugatePoint` analysis function to locate the image-space
  best-focus conjugate of an object-space point source.
- Added a `reference='ring'` option, along with `ring_radius` and `nring`
  arguments, to `wavefront`, `huygensPSF`, `fftPSF`, `spot`, `zernike`,
  `zernikeGQ`, `zernikeTA`, and `zernikeXYAberrations`.  This centers the
  output on the mean intersection of a ring of (possibly vignetted) rays,
  which is more robust than the chief ray for heavily vignetted fields.
- Added `parent` attribute for optics, so that non-sequential items co-move
  with a designated sibling under `withGloballyShiftedOptic`,
  `withLocallyRotatedOptic`, and `withGloballyRotatedOptic`.  Both short and
  fully qualified parent names are accepted, and `withRemovedOptic` now warns
  when it orphans an item.
- Added `extend` and `vignette_kw` arguments to `drawTrace2d` and
  `drawTrace3d`.
- Added `CoordSys.euler()` to recover intrinsic XYZ Tait-Bryan Euler angles.
- Added LTS-213 LSST optics description, including a CameraBody obscuration.
- Added Rubin as-built v1000 telescope description yamls.


Performance Improvements
------------------------


Bug Fixes
---------
- Fixed error in `traceFull` when both `path` and `reverse` were provided.
- Fixed error in `withRemovedOptic`.
- Fixed aliasing bugs in the `RayVector` factories: `fromStop` mutated
  user-supplied `x` and `y` arrays in place, and `asSpokes` returned a
  read-only `flux` array.
