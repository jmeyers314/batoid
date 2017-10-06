#ifndef batoid_coordsys_h
#define batoid_coordsys_h

#include "vec3.h"
#include "ray.h"
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace batoid {
    struct CoordSys {
        // explicitly construct the global coordinate system
        CoordSys();

        // _origin indicates the origin in global coordinates of the new coordinate system.
        // _rotation indicates the rotation applied to the global unit vectors to produce
        // the unit vectors of the new coordinate system.
        CoordSys(Vec3 _origin, Rot3 _rotation);

        // Could add an Euler angle ctor too...

        // Create a new CoordSys with parallel axes, but with origin shifted by dr,
        // where dr is in global coordinates.
        CoordSys shiftGlobal(const Vec3& dr) const;

        // Same, but here dr is in the local coordinate system.
        CoordSys shiftLocal(const Vec3& dr) const;

        // Rotate wrt the global axes, where the center of rotation is the global origin.
        CoordSys rotateGlobal(const Rot3& rotation) const;

        // Rotate wrt the global axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateGlobal(const Rot3& rotation, const Vec3& rotOrigin, std::shared_ptr<CoordSys> coordSys) const;

        // Rotate wrt the local axes, around the local origin
        CoordSys rotateLocal(const Rot3& rotation) const;

        // Rotate wrt the local axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateLocal(const Rot3& rotation, const Vec3& rotOrigin, std::shared_ptr<CoordSys> coordSys) const;

        // Get local unit vectors in global coordinates.
        Vec3 getX() const;
        Vec3 getY() const;
        Vec3 getZ() const;

        std::string repr() const {
            std::ostringstream oss(" ");
            oss << "CoordSys(" << origin << ", " << rotation << ')';
            return oss.str();
        }

        const Vec3 origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        const Rot3 rotation;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);

    struct CoordTransform {
        CoordTransform(std::shared_ptr<const CoordSys> source, std::shared_ptr<const CoordSys> destination);

        Vec3 applyForward(const Vec3& r) const;
        Vec3 applyReverse(const Vec3& r) const;

        Ray applyForward(const Ray& r) const;
        Ray applyReverse(const Ray& r) const;
        std::vector<Ray> applyForward(const std::vector<Ray>& r) const;
        std::vector<Ray> applyReverse(const std::vector<Ray>& r) const;

        const std::shared_ptr<const CoordSys> source;
        const std::shared_ptr<const CoordSys> destination;
        const Rot3 rotation;
        const Vec3 dr;
    };
}

#endif
