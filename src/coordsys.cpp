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
        CoordTransform toGlobal(coordSys, CoordSys());
        Vector3d globalRotCenter = toGlobal.applyForward(rotCenter);
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
        CoordTransform toGlobal(coordSys, CoordSys());
        Vector3d globalRotCenter = toGlobal.applyForward(rotCenter);
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


    // x is global coordinate
    // y is destination, with corresponding R and dr
    // z is source, with corresponding S and ds
    //
    // y = Rinv(x-dr)
    // z = Sinv(x-ds)
    // implies
    // x = S z + ds
    //
    // y = Rinv(S z + ds - dr)
    //   = Rinv S z + Rinv ds - Rinv dr
    //   = Rinv S z + Rinv S Sinv ds - Rinv S Sinv dr
    //   = Rinv S (z + Sinv ds - Sinv dr)
    //   = (Sinv R)^-1 (z - (Sinv dr - Sinv ds))
    //   = (Sinv R)^-1 (z - Sinv (dr - ds))

    CoordTransform::CoordTransform(const CoordSys& source, const CoordSys& destination) :
        _dr(source.m_rot.transpose()*(destination.m_origin - source.m_origin)),
        _rot(source.m_rot.transpose()*destination.m_rot),
        _source(source), _destination(destination) {}

    CoordTransform::CoordTransform(const Vector3d& dr, const Matrix3d& rot) :
        _dr(dr), _rot(rot) {}

    // We actively shift and rotate the coordinate system axes,
    // This looks like y = R x + dr
    // For a passive transformation of a fixed vector from one coord sys to another
    // though, we want the opposite transformation: y = R^-1 (x - dr)
    Vector3d CoordTransform::applyForward(const Vector3d& r) const {
        return _rot.transpose()*(r-_dr);
    }

    Vector3d CoordTransform::applyReverse(const Vector3d& r) const {
        return _rot*r+_dr;
    }

    Ray CoordTransform::applyForward(const Ray& r) const {
        if (r.failed) return r;
        return Ray(_rot.transpose()*(r.p0-_dr), _rot.transpose()*r.v,
                r.t0, r.wavelength, r.isVignetted);
    }

    Ray CoordTransform::applyReverse(const Ray& r) const {
        if (r.failed) return r;
        return Ray(_rot*r.p0 + _dr, _rot*r.v,
            r.t0, r.wavelength, r.isVignetted);
    }

    void CoordTransform::applyForwardInPlace(Ray& r) const {
        if (r.failed) return;
        r.p0 = _rot.transpose()*(r.p0-_dr);
        r.v = _rot.transpose()*r.v;
    }

    void CoordTransform::applyReverseInPlace(Ray& r) const {
        if (r.failed) return;
        r.p0 = _rot*r.p0+_dr;
        r.v = _rot*r.v;
    }

    std::vector<Ray> CoordTransform::applyForward(const std::vector<Ray>& rs) const {
        std::vector<Ray> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Ray& r) { return applyForward(r); }
        );
        return result;
    }

    std::vector<Ray> CoordTransform::applyReverse(const std::vector<Ray>& rs) const {
        std::vector<Ray> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Ray& r) { return applyReverse(r); }
        );
        return result;
    }

    void CoordTransform::applyForwardInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyForwardInPlace(r); }
        );
    }

    void CoordTransform::applyReverseInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyReverseInPlace(r); }
        );
    }

    bool operator==(const CoordTransform& ct1, const CoordTransform& ct2) {
        return ct1.getRot() == ct2.getRot() &&
               ct1.getDr() == ct2.getDr();
    }

    bool operator!=(const CoordTransform& ct1, const CoordTransform& ct2) {
        return !(ct1 == ct2);
    }

    std::ostream& operator<<(std::ostream &os, const CoordTransform& ct) {
        return os << ct.repr();
    }

}
