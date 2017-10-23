#ifndef batoid_vec3_h
#define batoid_vec3_h

#include <cmath>
#include <string>
#include <sstream>
#include <array>

namespace batoid {

    struct Vec3 {
        double x;
        double y;
        double z;

        Vec3() = default;

        Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

        Vec3(std::array<double,3> a) : x(a[0]), y(a[1]), z(a[2]) {}

        const double MagnitudeSquared() const {
            return x*x + y*y + z*z;
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

    inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        return os << v.repr();
    }

    inline Vec3 operator+(const Vec3& a, const Vec3& b) {
        return Vec3(a.x+b.x, a.y+b.y, a.z+b.z);
    }

    inline Vec3 operator-(const Vec3& a, const Vec3& b) {
        return Vec3(a.x-b.x, a.y-b.y, a.z-b.z);
    }

    inline Vec3 operator-(const Vec3& a) {
        return Vec3(-a.x, -a.y, -a.z);
    }

    inline Vec3 CrossProduct(const Vec3& a, const Vec3& b) {
        return Vec3(
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x));
    }

    inline double DotProduct(const Vec3& a, const Vec3& b) {
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

    // Should really be some sanity checks here that this is really a rotation matrix...
    struct Rot3 {
        std::array<double,9> data;

        // Default constructor makes identity rotation matrix.
        Rot3() : data({{1,0,0,
                        0,1,0,
                        0,0,1}}) {}
        Rot3(std::array<double,9> d) : data(d) {}
        Rot3(double thx, double thy, double thz) {
            double cx = cos(thx), sx = sin(thx);
            double cy = cos(thy), sy = sin(thy);
            double cz = cos(thz), sz = sin(thz);
            data = {{cy*cz,
                     -cy*sz,
                     sy,
                     cx*sz + sx*sy*cz,
                     cx*cz - sx*sy*sz,
                     -sx*cy,
                     sx*sz-cx*sy*cz,
                     sx*cz+cx*sy*sz,
                     cx*cy}};
        }

        Rot3(std::array<double,3> euler) : Rot3(euler[0], euler[1], euler[2]) {}

        std::array<double,3> getEuler() const {
            return {{atan2(-data[5], data[8]),
                     asin(data[2]),
                     atan2(-data[1], data[0])}};
        }


        inline double determinant() const {
            return data[0]*data[4]*data[8]
                 + data[1]*data[5]*data[6]
                 + data[2]*data[3]*data[7]
                 - data[0]*data[5]*data[7]
                 - data[1]*data[3]*data[8]
                 - data[2]*data[4]*data[6];
        }

        inline Rot3 inverse() const {
            return Rot3{{{data[0], data[3], data[6],
                          data[1], data[4], data[7],
                          data[2], data[5], data[8]}}};
        }

        std::string repr() const {
            std::ostringstream oss("\n");
            oss << "Rot3([" << data[0] << ", " << data[1] << ", " << data[2] << ", "
                            << data[3] << ", " << data[4] << ", " << data[5] << ", "
                            << data[6] << ", " << data[7] << ", " << data[8] << "])";
            return oss.str();
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Rot3& r) {
        return os << r.repr();
    }

    inline Vec3 RotVec(const Rot3& r, const Vec3& v) {
        return Vec3(r.data[0]*v.x + r.data[1]*v.y + r.data[2]*v.z,
                    r.data[3]*v.x + r.data[4]*v.y + r.data[5]*v.z,
                    r.data[6]*v.x + r.data[7]*v.y + r.data[8]*v.z);
    }

    inline Vec3 UnRotVec(const Rot3& r, const Vec3& v) {
        return Vec3(r.data[0]*v.x + r.data[3]*v.y + r.data[6]*v.z,
                    r.data[1]*v.x + r.data[4]*v.y + r.data[7]*v.z,
                    r.data[2]*v.x + r.data[5]*v.y + r.data[8]*v.z);
    }

    inline Rot3 operator*(const Rot3& a, const Rot3& b) {
        return Rot3{{{a.data[0]*b.data[0] + a.data[1]*b.data[3] + a.data[2]*b.data[6],
                      a.data[0]*b.data[1] + a.data[1]*b.data[4] + a.data[2]*b.data[7],
                      a.data[0]*b.data[2] + a.data[1]*b.data[5] + a.data[2]*b.data[8],

                      a.data[3]*b.data[0] + a.data[4]*b.data[3] + a.data[5]*b.data[6],
                      a.data[3]*b.data[1] + a.data[4]*b.data[4] + a.data[5]*b.data[7],
                      a.data[3]*b.data[2] + a.data[4]*b.data[5] + a.data[5]*b.data[8],

                      a.data[6]*b.data[0] + a.data[7]*b.data[3] + a.data[8]*b.data[6],
                      a.data[6]*b.data[1] + a.data[7]*b.data[4] + a.data[8]*b.data[7],
                      a.data[6]*b.data[2] + a.data[7]*b.data[5] + a.data[8]*b.data[8]
        }}};
    }

    inline bool operator==(const Rot3& a, const Rot3& b) {
        return a.data == b.data;
    }

    inline bool operator!=(const Rot3& a, const Rot3& b) {
        return a.data != b.data;
    }

}

#endif
