#include "coordSys.h"

namespace batoid {
    #pragma omp declare target
    CoordSys::CoordSys() :
        m_origin({0,0,0}),
        m_rot({1,0,0,  0,1,0,  0,0,1})
    {}

    CoordSys::CoordSys(const CoordSys& coordSys) :
        m_origin(coordSys.m_origin),
        m_rot(coordSys.m_rot)
    {}

    CoordSys::CoordSys(const vec3& origin, const mat3& rot) :
        m_origin(origin),
        m_rot(rot)
    {}

    CoordSys::CoordSys(const vec3& origin) :
        m_origin(origin),
        m_rot({1,0,0,  0,1,0,  0,0,1})
    {}

    CoordSys::CoordSys(const mat3& rot) :
        m_origin({0,0,0}),
        m_rot(rot)
    {}
    #pragma omp end declare target

    CoordSys CoordSys::shiftGlobal(const vec3& dr) const {
        return CoordSys(m_origin+dr, m_rot);
    }

    CoordSys CoordSys::shiftLocal(const vec3& dr) const {
        // Rotate the shift into global coordinates, then do the shift globally
        return shiftGlobal(m_rot*dr);
    }

    CoordSys CoordSys::rotateGlobal(const mat3& rot) const {
        // Want to rotate the current unit vectors expressed in m_rot.  So this
        // is just left matrix multiplication.
        // We rotate the origin vectors the same way.
        return CoordSys(rot*m_origin, rot*m_rot);
    }

    CoordSys CoordSys::rotateGlobal(
        const mat3& rot, const vec3& rotCenter, const CoordSys& coordSys
    ) const {
        vec3 globalRotCenter = coordSys.m_rot*rotCenter + coordSys.m_origin;
        return CoordSys(
            rot*(m_origin-globalRotCenter)+globalRotCenter,
            rot*m_rot
        );
    }

    CoordSys CoordSys::rotateLocal(const mat3& rot) const {
        // first rotate rot into global coords, then apply that
        // m_rot rot m_rot^-1 m_rot = m_rot rot
        return CoordSys(m_origin, m_rot*rot);
    }

    CoordSys CoordSys::rotateLocal(
        const mat3& rot, const vec3& rotCenter, const CoordSys& coordSys
    ) const {
        vec3 globalRotCenter = coordSys.m_rot*rotCenter + coordSys.m_origin;
        return CoordSys(
            m_rot*rot*transpose(m_rot)*(m_origin-globalRotCenter)+globalRotCenter,
            m_rot*rot
        );
    }

    vec3 CoordSys::getXHat() const {
        return {m_rot[0], m_rot[3], m_rot[6]};
    }

    vec3 CoordSys::getYHat() const {
        return {m_rot[1], m_rot[4], m_rot[7]};
    }

    vec3 CoordSys::getZHat() const {
        return {m_rot[2], m_rot[5], m_rot[8]};
    }

    bool operator==(const CoordSys& cs1, const CoordSys& cs2) {
        return cs1.m_origin == cs2.m_origin && cs1.m_rot == cs2.m_rot;
    }

    bool operator!=(const CoordSys& cs1, const CoordSys& cs2) {
        return !(cs1 == cs2);
    }

}
