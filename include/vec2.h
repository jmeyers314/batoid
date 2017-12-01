#ifndef batoid_vec2_h
#define batoid_vec2_h

#include <cmath>
#include <string>
#include <sstream>
#include <array>

namespace batoid {
    struct Vec2 {
        double x;
        double y;

        Vec2() = default;

        Vec2(double _x, double _y) : x(_x), y(_y) { }

        Vec2(std::array<double,2> a) : x(a[0]), y(a[1]) { }

        const double MagnitudeSquared() const {
            return x*x + y*y;
        }

        const double Magnitude() const {
            return std::hypot(x, y);
        }

        const Vec2 UnitVec2() const {
            const double mag = Magnitude();
            return Vec2(x/mag, y/mag);
        }

        Vec2& operator*=(const double factor) {
            x *= factor;
            y *= factor;
            return *this;
        }

        Vec2& operator+=(const Vec2& other) {
            x += other.x;
            y += other.y;
            return *this;
        }

        Vec2& operator-=(const Vec2& other) {
            x -= other.x;
            y -= other.y;
            return *this;
        }

        Vec2& operator/=(const double factor) {
            x /= factor;
            y /= factor;
            return *this;
        }

        std::string repr() const {
            std::ostringstream oss(" ");
            oss << "Vec2(" << x << ", " << y << ")";
            return oss.str();
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Vec2& v) {
        return os << v.repr();
    }

    inline Vec2 operator+(const Vec2& a, const Vec2& b) {
        return Vec2(a.x+b.x, a.y+b.y);
    }

    inline Vec2 operator-(const Vec2& a, const Vec2& b) {
        return Vec2(a.x-b.x, a.y-b.y);
    }

    inline Vec2 operator-(const Vec2& a) {
        return Vec2(-a.x, -a.y);
    }

    inline double DotProduct(const Vec2& a, const Vec2& b) {
        return a.x*b.x + a.y*b.y;
    }

    inline Vec2 operator*(double s, const Vec2& v) {
        return Vec2(s*v.x, s*v.y);
    }

    inline Vec2 operator*(const Vec2& v, double s) {
        return Vec2(s*v.x, s*v.y);
    }

    inline Vec2 operator/(const Vec2& v, double s) {
        return Vec2(v.x/s, v.y/s);
    }

    inline bool operator==(const Vec2& a, const Vec2& b) {
        return a.x == b.x && a.y == b.y;
    }

    inline bool operator!=(const Vec2& a, const Vec2& b) {
        return a.x != b.x || a.y != b.y;
    }

    // Should really be some sanity checks here that this is really a rotation matrix...
    struct Rot2 {
        std::array<double,4> data;

        // Default constructor makes identity rotation matrix.
        Rot2() : data({{1,0,
                        0,1}}) {}
        Rot2(std::array<double,4> d) : data(d) {}

        inline double determinant() const {
            return data[0]*data[3]
                 - data[1]*data[2];
        }

        std::string repr() const {
            std::ostringstream oss("\n");
            oss << "Rot2([" << data[0] << ", " << data[1] << ",\n"
                << "      " << data[2] << ", " << data[3] << "])\n";
            return oss.str();
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Rot2& r) {
        return os << r.repr();
    }

    inline Vec2 RotVec(const Rot2& r, const Vec2& v) {
        return Vec2(r.data[0]*v.x + r.data[1]*v.y,
                    r.data[2]*v.x + r.data[3]*v.y);
    }

    inline Vec2 UnRotVec(const Rot2& r, const Vec2& v) {
        return Vec2(r.data[0]*v.x + r.data[2]*v.y,
                    r.data[1]*v.x + r.data[3]*v.y);
    }

    inline bool operator==(const Rot2& r1, const Rot2& r2) {
        return r1.data == r2.data;
    }

    inline bool operator!=(const Rot2& r1, const Rot2& r2) {
        return r1.data != r2.data;
    }
}

#endif
