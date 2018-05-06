#ifndef batoid_coordsys_h
#define batoid_coordsys_h

#include "ray.h"
#include "rayVector.h"
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace batoid {
    struct CoordSys {
        // explicitly construct the global coordinate system
        CoordSys();

        // origin indicates the origin in global coordinates of the new coordinate system.
        // rot indicates the rotation applied to the global unit vectors to produce
        // the unit vectors of the new coordinate system.
        CoordSys(Vector3d origin, Matrix3d rot);
        CoordSys(Vector3d origin);
        CoordSys(Matrix3d rot);

        // Could add an Euler angle ctor too...

        // Create a new CoordSys with parallel axes, but with origin shifted by dr,
        // where dr is in global coordinates.
        CoordSys shiftGlobal(const Vector3d& dr) const;

        // Same, but here dr is in the local coordinate system.
        CoordSys shiftLocal(const Vector3d& dr) const;

        // Rotate wrt the global axes, where the center of rotation is the global origin.
        CoordSys rotateGlobal(const Matrix3d& rot) const;

        // Rotate wrt the global axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateGlobal(const Matrix3d& rot, const Vector3d& rotCenter,
            const CoordSys& coordSys) const;

        // Rotate wrt the local axes, around the local origin
        CoordSys rotateLocal(const Matrix3d& rot) const;

        // Rotate wrt the local axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateLocal(const Matrix3d& rot, const Vector3d& rotCenter,
            const CoordSys& coordSys) const;

        // Get coordSys unit vectors in global coordinates.
        Vector3d getXHat() const;
        Vector3d getYHat() const;
        Vector3d getZHat() const;

        std::string repr() const {
            std::ostringstream oss;
            oss << "CoordSys(" << m_origin << ", " << m_rot << ')';
            return oss.str();
        }

        const Vector3d m_origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        const Matrix3d m_rot;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordSys& cs1, const CoordSys& cs2);
    bool operator!=(const CoordSys& cs1, const CoordSys& cs2);


    class CoordTransform {
    public:
        CoordTransform(const CoordSys& source, const CoordSys& destination);
        CoordTransform(const Vector3d& dr, const Matrix3d& rot);

        Vector3d applyForward(const Vector3d& r) const;
        Vector3d applyReverse(const Vector3d& r) const;

        Ray applyForward(const Ray& r) const;
        Ray applyReverse(const Ray& r) const;

        void applyForwardInPlace(Ray& r) const;
        void applyReverseInPlace(Ray& r) const;

        std::vector<Ray> applyForward(const std::vector<Ray>& r) const;
        std::vector<Ray> applyReverse(const std::vector<Ray>& r) const;
        void applyForwardInPlace(std::vector<Ray>& rays) const;
        void applyReverseInPlace(std::vector<Ray>& rays) const;

        const Matrix3d& getRot() const { return _rot; }
        const Vector3d& getDr() const { return _dr; }

        std::string repr() const {
            std::ostringstream oss;
            oss << "CoordTransform(" << _source << ", " << _destination << ")";
            return oss.str();
        }

    private:
        const Vector3d _dr;
        const Matrix3d _rot;
        const CoordSys _source;
        const CoordSys _destination;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordTransform& ct1, const CoordTransform& ct2);
    bool operator!=(const CoordTransform& ct1, const CoordTransform& ct2);
}

#endif
