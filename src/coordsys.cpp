#include "coordsys.h"
#include "utils.h"
#include "vec3.h"
#include <iostream>

namespace batoid {

    CoordSys::CoordSys() :
        origin(Vec3()), rotation(Rot3()) {}

    CoordSys::CoordSys(const Vec3 _origin, const Rot3 _rotation) :
        origin(_origin), rotation(_rotation) {}

    CoordSys CoordSys::shiftGlobal(const Vec3& dr) const {
        return CoordSys(origin+dr, rotation);
    }

    CoordSys CoordSys::shiftLocal(const Vec3& dr) const {
        // Note RotVec instead of UnRotVec, b/c we are doing a passive rotation
        // instead of an active one.
        return shiftGlobal(RotVec(rotation, dr));
    }

    CoordSys CoordSys::rotateGlobal(const Rot3& _rotation) const {
        return CoordSys(RotVec(_rotation, origin), _rotation*rotation);
    }

    CoordSys CoordSys::rotateGlobal(const Rot3& _rotation, const Vec3& rotCenter, const CoordSys& coordSys) const {
        CoordTransform toGlobal(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        return CoordSys(
            RotVec(_rotation, origin-globalRotCenter)+globalRotCenter,
            _rotation*rotation
        );
    }

    CoordSys CoordSys::rotateLocal(const Rot3& _rotation) const {
        return CoordSys(
            origin,
            rotation.inverse()*_rotation*rotation
        );
    }

    CoordSys CoordSys::rotateLocal(const Rot3& _rotation, const Vec3& rotCenter, const CoordSys& coordSys) const {
        CoordTransform toGlobal(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        std::cout << "globalRotCenter = " << globalRotCenter << '\n';
        std::cout << "origin = " << origin << '\n';
        return CoordSys(
            RotVec(_rotation, origin-globalRotCenter),
            rotation.inverse()*_rotation*rotation
        );
    }

    Vec3 CoordSys::getXHat() const {
        return Vec3(rotation.data[0], rotation.data[3], rotation.data[6]);
    }

    Vec3 CoordSys::getYHat() const {
        return Vec3(rotation.data[1], rotation.data[4], rotation.data[7]);
    }

    Vec3 CoordSys::getZHat() const {
        return Vec3(rotation.data[2], rotation.data[5], rotation.data[8]);
    }

    std::ostream& operator<<(std::ostream& os, const CoordSys& cs) {
        return os << cs.repr();
    }

    bool operator==(const CoordSys& cs1, const CoordSys& cs2) {
        return cs1.origin == cs2.origin && cs1.rotation == cs2.rotation;
    }

    bool operator!=(const CoordSys& cs1, const CoordSys& cs2) {
        return !(cs1 == cs2);
    }


    // x is global
    // y is destination, with corresponding R and dr
    // z is source, with corresponding S and ds
    //
    // y = Rinv(x-dr)
    // z = Sinv(x-ds)
    // x = S z + ds
    //
    // y = Rinv(S z + ds - dr)
    //   = Rinv S z + Rinv ds - Rinv dr
    //   = Rinv S z + Rinv S Sinv ds - Rinv S Sinv dr
    //   = Rinv S (z + Sinv ds - Sinv dr)
    //   = (Sinv R)^-1 (z - (Sinv dr - Sinv ds))
    //   = (Sinv R)^-1 (z - Sinv (dr - ds))

    CoordTransform::CoordTransform(const CoordSys& source, const CoordSys& destination) :
        _dr(UnRotVec(source.rotation, destination.origin - source.origin)),
        _rot(source.rotation.inverse()*destination.rotation),
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
            [this](const Ray& r) { return applyForward(r); },
            2000
        );
        return result;
    }

    std::vector<Ray> CoordTransform::applyReverse(const std::vector<Ray>& rs) const {
        std::vector<Ray> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Ray& r) { return applyReverse(r); },
            2000
        );
        return result;
    }

    void CoordTransform::applyForwardInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyForwardInPlace(r); },
            2000
        );
    }

    void CoordTransform::applyReverseInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyReverseInPlace(r); },
            2000
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
