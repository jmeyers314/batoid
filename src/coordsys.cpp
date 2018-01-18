#include "coordsys.h"
#include "utils.h"
#include "vec3.h"
#include <iostream>

namespace batoid {

    CoordSys::CoordSys() :
        m_origin(Vec3()), m_rot(Rot3()) {}

    CoordSys::CoordSys(const Vec3 origin, const Rot3 rot) :
        m_origin(origin), m_rot(rot) {}

    CoordSys::CoordSys(const Vec3 origin) :
        m_origin(origin), m_rot(Rot3()) {}

    CoordSys::CoordSys(const Rot3 rot) :
        m_origin(Vec3()), m_rot(rot) {}

    CoordSys CoordSys::shiftGlobal(const Vec3& dr) const {
        return CoordSys(m_origin+dr, m_rot);
    }

    CoordSys CoordSys::shiftLocal(const Vec3& dr) const {
        // Note RotVec instead of UnRotVec, b/c we are doing a passive rotation
        // instead of an active one.
        return shiftGlobal(RotVec(m_rot, dr));
    }

    CoordSys CoordSys::rotateGlobal(const Rot3& rot) const {
        return CoordSys(RotVec(rot, m_origin), rot*m_rot);
    }

    CoordSys CoordSys::rotateGlobal(const Rot3& rot, const Vec3& rotCenter, const CoordSys& coordSys) const {
        CoordTransform toGlobal(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        return CoordSys(
            RotVec(rot, m_origin-globalRotCenter)+globalRotCenter,
            rot*m_rot
        );
    }

    CoordSys CoordSys::rotateLocal(const Rot3& rot) const {
        // first rotate rot into global coords, then apply that
        // m_rot rot m_rot^-1 m_rot = m_rot rot
        return CoordSys(m_origin, m_rot*rot);
    }

    CoordSys CoordSys::rotateLocal(const Rot3& rot, const Vec3& rotCenter, const CoordSys& coordSys) const {
        CoordTransform toGlobal(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        return CoordSys(
            RotVec(m_rot*rot*m_rot.inverse(), m_origin-globalRotCenter)+globalRotCenter,
            m_rot*rot
        );
    }

    Vec3 CoordSys::getXHat() const {
        return Vec3(m_rot.data[0], m_rot.data[3], m_rot.data[6]);
    }

    Vec3 CoordSys::getYHat() const {
        return Vec3(m_rot.data[1], m_rot.data[4], m_rot.data[7]);
    }

    Vec3 CoordSys::getZHat() const {
        return Vec3(m_rot.data[2], m_rot.data[5], m_rot.data[8]);
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
        _dr(UnRotVec(source.m_rot, destination.m_origin - source.m_origin)),
        _rot(source.m_rot.inverse()*destination.m_rot),
        _source(source), _destination(destination) {}

    CoordTransform::CoordTransform(const Vec3& dr, const Rot3& rot) :
        _dr(dr), _rot(rot) {}

    // We actively shift and rotate the coordinate system axes,
    // This looks like y = R x + dr
    // For a passive transformation of a fixed vector from one coord sys to another
    // though, we want the opposite transformation: y = R^-1 (x - dr)
    Vec3 CoordTransform::applyForward(const Vec3& r) const {
        return UnRotVec(_rot, r-_dr);
    }

    Vec3 CoordTransform::applyReverse(const Vec3& r) const {
        return RotVec(_rot, r)+_dr;
    }

    Ray CoordTransform::applyForward(const Ray& r) const {
        if (r.failed) return r;
        return Ray(UnRotVec(_rot, r.p0-_dr), UnRotVec(_rot, r.v),
                r.t0, r.wavelength, r.isVignetted);
    }

    Ray CoordTransform::applyReverse(const Ray& r) const {
        if (r.failed) return r;
        return Ray(RotVec(_rot, r.p0) + _dr, RotVec(_rot, r.v),
            r.t0, r.wavelength, r.isVignetted);
    }

    void CoordTransform::applyForwardInPlace(Ray& r) const {
        if (r.failed) return;
        r.p0 = UnRotVec(_rot, r.p0-_dr);
        r.v = UnRotVec(_rot, r.v);
    }

    void CoordTransform::applyReverseInPlace(Ray& r) const {
        if (r.failed) return;
        r.p0 = RotVec(_rot, r.p0)+_dr;
        r.v = RotVec(_rot, r.v);
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
