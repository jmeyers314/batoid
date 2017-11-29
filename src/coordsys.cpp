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
        auto toGlobal = getTransform(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal->applyForward(rotCenter);
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
        auto toGlobal = getTransform(coordSys, CoordSys());
        Vec3 globalRotCenter = toGlobal->applyForward(rotCenter);
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

    std::vector<Ray> BaseCoordTransform::applyForward(const std::vector<Ray>& rs) const {
        std::vector<Ray> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Ray& r) { return applyForward(r); },
            2000
        );
        return result;
    }

    std::vector<Ray> BaseCoordTransform::applyReverse(const std::vector<Ray>& rs) const {
        std::vector<Ray> result(rs.size());
        parallelTransform(rs.cbegin(), rs.cend(), result.begin(),
            [this](const Ray& r) { return applyReverse(r); },
            2000
        );
        return result;
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

    std::unique_ptr<BaseCoordTransform> getTransform(const CoordSys& source, const CoordSys& destination) {
        Rot3 rot = source.rotation.inverse()*destination.rotation;
        Vec3 dr = UnRotVec(source.rotation, destination.origin - source.origin);
        if (rot == Rot3()) {
            if (dr == Vec3()) {
                return std::unique_ptr<BaseCoordTransform>(new IdentityTransform());
            } else {
                return std::unique_ptr<BaseCoordTransform>(new ShiftTransform(dr));
            }
        } else {
            if (dr == Vec3()) {
                return std::unique_ptr<BaseCoordTransform>(new RotTransform(rot));
            } else {
                return std::unique_ptr<BaseCoordTransform>(new CoordTransform(dr, rot));
            }
        }
    }

    void BaseCoordTransform::applyForwardInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyForwardInPlace(r); },
            2000
        );
    }

    void BaseCoordTransform::applyReverseInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(rays.begin(), rays.end(),
            [this](Ray& r) { applyReverseInPlace(r); },
            2000
        );
    }
}
