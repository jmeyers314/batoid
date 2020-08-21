#include "coordsys.h"
#include "utils.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace batoid {

    CoordSys::CoordSys() :
        m_origin(Vector3d::Zero()), m_rot(Matrix3d::Identity()) {}

    CoordSys::CoordSys(const Vector3d origin, const Matrix3d rot) :
        m_origin(origin), m_rot(rot) {}

    CoordSys::CoordSys(const Vector3d origin) :
        m_origin(origin), m_rot(Matrix3d::Identity()) {}

    CoordSys::CoordSys(const Matrix3d rot) :
        m_origin(Vector3d::Zero()), m_rot(rot) {}

    CoordSys CoordSys::shiftGlobal(const Vector3d& dr) const {
        return CoordSys(m_origin+dr, m_rot);
    }

    CoordSys CoordSys::shiftLocal(const Vector3d& dr) const {
        // Note m_rot*dr instead of m_rot.transpose()*dr, b/c we are doing a passive rotation
        // instead of an active one.
        return shiftGlobal(m_rot*dr);
    }

    CoordSys CoordSys::rotateGlobal(const Matrix3d& rot) const {
        return CoordSys(rot*m_origin, rot*m_rot);
    }

    CoordSys CoordSys::rotateGlobal(const Matrix3d& rot, const Vector3d& rotCenter, const CoordSys& coordSys) const {
        // Hard code the below to avoid circular include
        // CoordTransform toGlobal(coordSys, CoordSys());
        // Vector3d globalRotCenter = toGlobal.applyForward(rotCenter);
        Vector3d globalRotCenter = coordSys.m_rot*(rotCenter+coordSys.m_rot.transpose()*(coordSys.m_origin));
        return CoordSys(
            rot*(m_origin-globalRotCenter)+globalRotCenter,
            rot*m_rot
        );
    }

    CoordSys CoordSys::rotateLocal(const Matrix3d& rot) const {
        // first rotate rot into global coords, then apply that
        // m_rot rot m_rot^-1 m_rot = m_rot rot
        return CoordSys(m_origin, m_rot*rot);
    }

    CoordSys CoordSys::rotateLocal(const Matrix3d& rot, const Vector3d& rotCenter, const CoordSys& coordSys) const {
        // CoordTransform toGlobal(coordSys, CoordSys());
        // Vector3d globalRotCenter = toGlobal.applyForward(rotCenter);
        Vector3d globalRotCenter = coordSys.m_rot*(rotCenter+coordSys.m_rot.transpose()*(coordSys.m_origin));
        return CoordSys(
            m_rot*rot*(m_rot.transpose())*(m_origin-globalRotCenter)+globalRotCenter,
            m_rot*rot
        );
    }

    Vector3d CoordSys::getXHat() const {
        return Vector3d(m_rot.data()[0], m_rot.data()[3], m_rot.data()[6]);
    }

    Vector3d CoordSys::getYHat() const {
        return Vector3d(m_rot.data()[1], m_rot.data()[4], m_rot.data()[7]);
    }

    Vector3d CoordSys::getZHat() const {
        return Vector3d(m_rot.data()[2], m_rot.data()[5], m_rot.data()[8]);
    }

    std::ostream& operator<<(std::ostream& os, const CoordSys& cs) {
        return os << cs.repr();
    }

    bool operator==(const CoordSys& cs1, const CoordSys& cs2) {
        return cs1.m_origin == cs2.m_origin && cs1.m_rot == cs2.m_rot;
    }

    bool operator!=(const CoordSys& cs1, const CoordSys& cs2) {
        return !(cs1 == cs2);
    }

}
