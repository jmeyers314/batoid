#include "vecmath.h"

namespace batoid {
    // Assume mat3 is row-major.

    vec3 operator+(const vec3& lhs, const vec3& rhs) {
        return {lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]};
    }

    vec3 operator-(const vec3& lhs, const vec3& rhs) {
        return {lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]};
    }

    mat3 transpose(const mat3& M) {
        return {
            M[0], M[3], M[6],
            M[1], M[4], M[7],
            M[2], M[5], M[8]
        };
    }

    vec3 operator*(const mat3& M, const vec3& v) {
        double x = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
        double y = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
        double z = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
        return {x, y, z};
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
}
