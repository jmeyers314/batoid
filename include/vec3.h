#ifndef __jem_vec3__h
#define __jem_vec3__h

#include <cmath>
#include <string>
#include <sstream>

namespace jtrace {

    struct Vec3 {
        double x;
        double y;
        double z;

        Vec3() : x(0.0), y(0.0), z(0.0) {}

        Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

        const double MagnitudeSquared() const {
            return (x*x)+(y*y)+(z*z);
        }

        const double Magnitude() const {
            return std::sqrt(MagnitudeSquared());
        }

        const Vec3 UnitVec3() const {
            const double mag = Magnitude();
            return Vec3(x/mag, y/mag, z/mag);
        }

        Vec3& operator*=(const double factor) {
            x *= factor;
            y *= factor;
            z *= factor;
            return *this;
        }

        Vec3& operator+=(const Vec3& other) {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        Vec3& operator-=(const Vec3& other) {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        Vec3& operator/=(const double factor) {
            x /= factor;
            y /= factor;
            z /= factor;
            return *this;
        }

        std::string repr() const {
            std::ostringstream oss(" ");
            oss << "Vec3(" << x << ", " << y << ", " << z << ")";
            return oss.str();
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Vec3 &v) {
        return os << v.repr();
    }

    inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.x+b.x, a.y+b.y, a.z+b.z);
    }

    inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.x-b.x, a.y-b.y, a.z-b.z);
    }

    inline Vec3 operator-(const Vec3 &a) {
        return Vec3(-a.x, -a.y, -a.z);
    }

    inline Vec3 CrossProduct(const Vec3 &a, const Vec3 &b) {
        return Vec3(
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x));
    }

    inline double DotProduct(const Vec3 &a, const Vec3 &b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    inline Vec3 operator*(double s, const Vec3& v) {
        return Vec3(s*v.x, s*v.y, s*v.z);
    }

    inline Vec3 operator*(const Vec3& v, double s) {
        return Vec3(s*v.x, s*v.y, s*v.z);
    }

    inline Vec3 operator/(const Vec3& v, double s) {
        return Vec3(v.x/s, v.y/s, v.z/s);
    }

    inline bool operator==(const Vec3& a, const Vec3& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    inline bool operator!=(const Vec3& a, const Vec3& b) {
        return a.x != b.x || a.y != b.y || a.z != b.z;
    }
}

#endif
