Coordinate Systems
==================

Batoid uses coordinate systems to track the positions and orientations of Rays
and Surfaces.  By using coordinate systems, one only ever has to define how
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
