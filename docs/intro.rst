Introduction
============

Batoid is a c++-backed python optical raytracing program.  It's primary intended use is for analysis
of wide-field optical survey telescopes, like DECam, HSC, or LSST, although it is potentially useful
for a wide variety of optical systems.  The primary features include:

- geometric ray tracing through an optical system.
- ability to apply solid-body perturbations to elements of an optical system, hierarchically.
- wavefront / optical-path-difference calculations.
- diffraction calculations such as the Fourier optics PSF and Huygens construction PSF.

This guide contains a (very) brief introduction to some of these features.

Basic objects
-------------

The primary python classes in batoid are batoid.Ray and batoid.Surface.

batoid.Ray
~~~~~~~~~~

A batoid.Ray represents a single geometric ray that can be traced through an optical system.  A ray
has attributes including

- a position `.r`.  This is always a 3-vector measured in meters.
- a velocity `.v`.  This is a 3-vector and is measured in units of the speed of light.  I.e.,
  for a Ray in a vacuum, the norm of the velocity 3-vector is :math:`1`.  In a medium with
  refractive index :math:`n`, the norm will be :math:`1/n`.
- a time `.t`.  This is a reference time indicating when the ray was, is, or will be at the
  reference position `.r`.  Although this variable may be thought of as time, it's more precisely
  an optical path length.  I.e., it's the accumulated time multiplied by the speed-of-light. The
  units are meters.
- a wavelength `.wavelength`.  This is the vacuum wavelength in meters.
- a boolean indicator `.vignetted`, which indicates if the ray has been vignetted while being
  traced through an optical system.
- a boolean indicatory `.failed`, which indicates if some attempted calculation, like an
  intersection of a Ray with a Surface, failed.

The batoid.RayVector class also exists to handle rapid calculations involving more than one Ray.


batoid.Surface
~~~~~~~~~~~~~~

A batoid.Surface object represents a single 2-dimensional optical surface, such as the surface of a
mirror, one of the surfaces of a lens, a baffle, or a detector.  Surface is an abstract base class,
and as such cannot be instantiated directly.  A number of subclasses are available in batoid to
describe surfaces with different profiles `z(x, y)`.  Many of these are circularly symmetric, and
can be written as functions of :math:`r = \sqrt{x^2+y^2}`.

Plane
.....
  :math:`z = 0`.

  There are no parameters for a `batoid.Plane` object.

Paraboloid
..........
  :math:`z = \frac{r^2}{2R}`

  The parameter :math:`R` indicates the radius of curvature at the paraboloid vertex, which is the
  origin.

Sphere
......
  :math:`z = \frac{r^2}{R\left(1+\sqrt{1-r^2/R^2}\right)}`

  The parameter :math:`R` indicates the radius of curvature of the sphere.

Quadric
.......
  :math:`z = \frac{r^2}{R\left(1+\sqrt{1-(1+\kappa)r^2/R^2}\right)}`

  The parameter :math:`R` indicates the radius of curvature and quadric vertex, which is the
  origin, and :math:`\kappa` indicates the conic constant of the surface.

Asphere
.......
  :math:`z = \frac{r^2}{R\left(1+\sqrt{1-(1+\kappa)r^2/R^2}\right)} + \alpha_4 r^4 + \alpha_6 r^6 + ...`

  The parameters :math:`R` and :math:`\kappa` are the same as for Quadric.  The sequence of
  :math:`\{\alpha_4, \alpha_6, ...\}` are optional parameters indicating the departure from a
  quadric surface.

Zernike
.......
  :math:`z = \sum_j a_j Z_j(x, y)`

  The :math:`Z_j(x, y)` are the Zernike polynomials, indexed by the Noll convention.  The sequence
  :math:`\{a_j\}` are the coefficients for each polynomial.  This is currently the only way of
  specifying a non-circularly-symmetric surface.

Sum
...
  :math:`z = \sum_j z_j(x, y)`

  Used to construct surfaces that are the sums of other surfaces.


Todo
====

batoid.Optic
------------

batiod.Obscuration
------------------

batoid.CoordSys
---------------

batoid.CoordTransform
---------------------

batoid.Medium
-------------

yaml parser
-----------

ray generator
-------------
