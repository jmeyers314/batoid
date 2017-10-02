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

    CoordSys CoordSys::rotateGlobal(const Rot3& _rotation, const Vec3& rotCenter, std::shared_ptr<CoordSys> coordSys) const {
        CoordTransform toGlobal(coordSys, std::make_shared<CoordSys>(CoordSys()));
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

    CoordSys CoordSys::rotateLocal(const Rot3& _rotation, const Vec3& rotCenter, std::shared_ptr<CoordSys> coordSys) const {
        CoordTransform toGlobal(coordSys, std::make_shared<CoordSys>(CoordSys()));
        Vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        std::cout << "globalRotCenter = " << globalRotCenter << '\n';
        std::cout << "origin = " << origin << '\n';
        return CoordSys(
            RotVec(_rotation, origin-globalRotCenter),
            rotation.inverse()*_rotation*rotation
        );
    }

    Vec3 CoordSys::getX() const {
        return Vec3(rotation.data[0], rotation.data[3], rotation.data[6]);
    }

    Vec3 CoordSys::getY() const {
        return Vec3(rotation.data[1], rotation.data[4], rotation.data[7]);
    }

    Vec3 CoordSys::getZ() const {
        return Vec3(rotation.data[2], rotation.data[5], rotation.data[8]);
    }

    std::ostream& operator<<(std::ostream& os, const CoordSys& cs) {
        return os << cs.repr();
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
    //
    CoordTransform::CoordTransform(
        std::shared_ptr<const CoordSys> _source,
        std::shared_ptr<const CoordSys> _destination) :
        source(_source),
        destination(_destination),
        rotation(source->rotation.inverse()*destination->rotation),
        dr(UnRotVec(source->rotation, destination->origin - source->origin))
        {}

    // We actively shift and rotate the coordinate system axes,
    // This looks like y = R x + dr
    // For a passive transformation of a fixed vector from one coord sys to another
    // though, we want the opposite transformation: y = R^-1 (x - dr)
    Vec3 CoordTransform::applyForward(const Vec3& r) const {
        return UnRotVec(rotation, r-dr);
    }

    Vec3 CoordTransform::applyReverse(const Vec3& r) const {
        return RotVec(rotation, r)+dr;
    }

    std::vector<Vec3> CoordTransform::applyForward(const std::vector<Vec3>& rs) const {
        std::vector<Vec3> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Vec3& r) { return applyForward(r); },
            2000
        );
        return result;
    }

    std::vector<Vec3> CoordTransform::applyReverse(const std::vector<Vec3>& rs) const {
        std::vector<Vec3> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Vec3& r) { return applyReverse(r); },
            2000
        );
        return result;
    }

    Ray CoordTransform::applyForward(const Ray& r) const {
        return Ray(
            UnRotVec(rotation, r.p0-dr),
            UnRotVec(rotation, r.v),
            r.t0, r.wavelength, r.isVignetted
        );
    }

    Ray CoordTransform::applyReverse(const Ray& r) const {
        return Ray(
            RotVec(rotation, r.p0) + dr,
            RotVec(rotation, r.v),
            r.t0, r.wavelength, r.isVignetted
        );
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
}
