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
        CoordSys rotateGlobal(const Rot3& rotation, const Vec3& rotOrigin, const CoordSys& coordSys) const;

        // Rotate wrt the local axes, around the local origin
        CoordSys rotateLocal(const Rot3& rotation) const;

        // Rotate wrt the local axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateLocal(const Rot3& rotation, const Vec3& rotOrigin, const CoordSys& coordSys) const;

        // Get local unit vectors in global coordinates.
        Vec3 getXHat() const;
        Vec3 getYHat() const;
        Vec3 getZHat() const;

        std::string repr() const {
            std::ostringstream oss;
            oss << "CoordSys(" << origin << ", " << rotation << ')';
            return oss.str();
        }

        const Vec3 origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        const Rot3 rotation;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordSys& cs1, const CoordSys& cs2);
    bool operator!=(const CoordSys& cs1, const CoordSys& cs2);


    class CoordTransform {
    public:
        CoordTransform(const CoordSys& source, const CoordSys& destination);
        CoordTransform(const Vec3& dr, const Rot3& rot);

        Vec3 applyForward(const Vec3& r) const;
        Vec3 applyReverse(const Vec3& r) const;

        Ray applyForward(const Ray& r) const;
        Ray applyReverse(const Ray& r) const;

        void applyForwardInPlace(Ray& r) const;
        void applyReverseInPlace(Ray& r) const;

        std::vector<Ray> applyForward(const std::vector<Ray>& r) const;
        std::vector<Ray> applyReverse(const std::vector<Ray>& r) const;
        void applyForwardInPlace(std::vector<Ray>& rays) const;
        void applyReverseInPlace(std::vector<Ray>& rays) const;

        Rot3 getRot() const { return _rot; }
        Vec3 getDr() const { return _dr; }

        std::string repr() const {
            std::ostringstream oss;
            oss << "CoordTransform(" << _source << ", " << _destination << ")";
            return oss.str();
        }

    private:
        const Vec3 _dr;
        const Rot3 _rot;
        const CoordSys _source;
        const CoordSys _destination;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordTransform& ct1, const CoordTransform& ct2);
    bool operator!=(const CoordTransform& ct1, const CoordTransform& ct2);
}

#endif
