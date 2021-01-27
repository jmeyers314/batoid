Coordinate Systems
==================

Batoid uses coordinate systems to track the positions and orientations of rays
and surfaces.  By using coordinate systems, one only ever has to define how
surfaces behave when their vertexes are at the origin (0,0,0); relocating the
surface to a non-origin vertex is accomplished by transforming coordinates.

There are two parts to a coordinate system in batoid:

  - the origin, which specifies in global coordinates to the location of
    (0, 0, 0) in local coordinates, and
  - the orientation, which specifies how the local x, y, and z axes are rotated
    with respect to the global axes.

.. autoclass:: batoid.CoordSys
    :members:

.. autoclass:: batoid.CoordTransform
    :members:


Angle Projections
-----------------

For exact specification of ray directions, it's convenient to use a
two-dimensional projection.  Batoid includes a number of such projections and
deprojections:

.. autofunction:: batoid.utils.fieldToDirCos

.. autofunction:: batoid.utils.dirCosToField

.. autofunction:: batoid.utils.gnomonicToDirCos

.. autofunction:: batoid.utils.dirCosToGnomonic

.. autofunction:: batoid.utils.postelToDirCos

.. autofunction:: batoid.utils.dirCosToPostel

.. autofunction:: batoid.utils.zemaxToDirCos

.. autofunction:: batoid.utils.dirCosToZemax

.. autofunction:: batoid.utils.stereographicToDirCos

.. autofunction:: batoid.utils.dirCosToStereographic

.. autofunction:: batoid.utils.orthographicToDirCos

.. autofunction:: batoid.utils.dirCosToOrthographic

.. autofunction:: batoid.utils.lambertToDirCos

.. autofunction:: batoid.utils.dirCosToLambert
