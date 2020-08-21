#ifndef batoid_coordSys_h
#define batoid_coordSys_h

#include <array>

namespace batoid {
    struct CoordSys {
        using vec3 = std::array<double, 3>;
        using mat3 = std::array<double, 9>;

        // explicitly construct the global coordinate system
        CoordSys();
        // Copy-ctor
        CoordSys(const CoordSys& coordSys);

        // origin indicates the origin in global coordinates of the new coordinate system.
        // rot indicates the rotation applied to the global unit vectors to produce
        // the unit vectors of the new coordinate system.
        CoordSys(const vec3& origin, const mat3& rot);
        CoordSys(const vec3& origin);
        CoordSys(const mat3& rot);

        // Could add an Euler angle ctor too...

        // Create a new CoordSys with parallel axes, but with origin shifted by dr,
        // where dr is in global coordinates.
        CoordSys shiftGlobal(const vec3& dr) const;

        // Same, but here dr is in the local coordinate system.
        CoordSys shiftLocal(const vec3& dr) const;

        // Rotate wrt the global axes, where the center of rotation is the global origin.
        CoordSys rotateGlobal(const mat3& rot) const;

        // Rotate wrt the global axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateGlobal(
            const mat3& rot, const vec3& rotCenter,
            const CoordSys& coordSys
        ) const;

        // Rotate wrt the local axes, around the local origin
        CoordSys rotateLocal(const mat3& rot) const;

        // Rotate wrt the local axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateLocal(
            const mat3& rot, const vec3& rotCenter,
            const CoordSys& coordSys
        ) const;

        // Get coordSys unit vectors in global coordinates.
        vec3 getXHat() const;
        vec3 getYHat() const;
        vec3 getZHat() const;

        vec3 m_origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        mat3 m_rot;
    };

    bool operator==(const CoordSys& cs1, const CoordSys& cs2);
    bool operator!=(const CoordSys& cs1, const CoordSys& cs2);
}

#endif
