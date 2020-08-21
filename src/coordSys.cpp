#include "coordSys.h"

namespace batoid {
    using vec3 = CoordSys::vec3;
    using mat3 = CoordSys::mat3;

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

    vec3 operator*(const mat3& M, const vec3& v) {
        double x = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
        double y = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
        double z = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
        return {x, y, z};
    }

    vec3 operator+(const vec3& lhs, const vec3& rhs) {
        return {lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]};
    }

    vec3 operator-(const vec3& lhs, const vec3& rhs) {
        return {lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]};
    }

    mat3 operator*(const mat3& lhs, const mat3& rhs) {
        return {
            lhs[0]*rhs[0] + lhs[1]*rhs[3] + lhs[2]*rhs[6],
            lhs[0]*rhs[1] + lhs[1]*rhs[4] + lhs[2]*rhs[7],
            lhs[0]*rhs[2] + lhs[1]*rhs[5] + lhs[2]*rhs[8],

            lhs[3]*rhs[0] + lhs[4]*rhs[3] + lhs[5]*rhs[6],
            lhs[3]*rhs[1] + lhs[4]*rhs[4] + lhs[5]*rhs[7],
            lhs[3]*rhs[2] + lhs[4]*rhs[5] + lhs[5]*rhs[8],

            lhs[6]*rhs[0] + lhs[7]*rhs[3] + lhs[8]*rhs[6],
            lhs[6]*rhs[1] + lhs[7]*rhs[4] + lhs[8]*rhs[7],
            lhs[6]*rhs[2] + lhs[7]*rhs[5] + lhs[8]*rhs[8]
        };
    }

    mat3 transpose(const mat3& M) {
        return {
            M[0], M[3], M[6],
            M[1], M[4], M[7],
            M[2], M[5], M[8]
        };
    }

    CoordSys CoordSys::shiftGlobal(const vec3& dr) const {
        return CoordSys(m_origin+dr, m_rot);
    }

    CoordSys CoordSys::shiftLocal(const vec3& dr) const {
        // Note m_rot*dr instead of m_rot.transpose()*dr, b/c we are doing a passive rotation
        // instead of an active one.
        return shiftGlobal(m_rot*dr);
    }

    CoordSys CoordSys::rotateGlobal(const mat3& rot) const {
        return CoordSys(rot*m_origin, rot*m_rot);
    }

    CoordSys CoordSys::rotateGlobal(
        const mat3& rot, const vec3& rotCenter, const CoordSys& coordSys
    ) const {
        // Hard code the below to avoid circular include
        // CoordTransform toGlobal(coordSys, CoordSys());
        // vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        vec3 globalRotCenter = coordSys.m_rot*(
            rotCenter+transpose(coordSys.m_rot)*(coordSys.m_origin)
        );
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
        // CoordTransform toGlobal(coordSys, CoordSys());
        // vec3 globalRotCenter = toGlobal.applyForward(rotCenter);
        vec3 globalRotCenter = coordSys.m_rot*(
            rotCenter+transpose(coordSys.m_rot)*(coordSys.m_origin)
        );
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
