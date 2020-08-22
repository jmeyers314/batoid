#ifndef batoid_math_h
#define batoid_math_h

#include <array>

namespace batoid {
    using vec3 = std::array<double, 3>;
    using mat3 = std::array<double, 9>;

    vec3 operator*(const mat3& M, const vec3& v);

    vec3 operator+(const vec3& lhs, const vec3& rhs);

    vec3 operator-(const vec3& lhs, const vec3& rhs);

    mat3 operator*(const mat3& lhs, const mat3& rhs);

    mat3 transpose(const mat3& M);
}

#endif
