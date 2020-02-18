#ifndef batoid_coordsys_h
#define batoid_coordsys_h

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
        CoordSys(const CoordSys& coordSys) : m_origin(coordSys.m_origin), m_rot(coordSys.m_rot) {}

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
            oss << "CoordSys([" << m_origin[0] << "," << m_origin[1] << "," << m_origin[2] << "],[["
                << m_rot(0) << "," << m_rot(3) << "," << m_rot(6) << "],["
                << m_rot(1) << "," << m_rot(4) << "," << m_rot(7) << "],["
                << m_rot(2) << "," << m_rot(5) << "," << m_rot(8) << "]])";
            return oss.str();
        }

        Vector3d m_origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        Matrix3d m_rot;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordSys& cs1, const CoordSys& cs2);
    bool operator!=(const CoordSys& cs1, const CoordSys& cs2);
}

#endif
