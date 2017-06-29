#ifndef jtrace_ray_h
#define jtrace_ray_h

#include <sstream>
#include <string>
#include "vec3.h"

namespace jtrace {
    struct Ray {
        Ray(double x0, double y0, double z0, double vx, double vy, double vz,
            double t, double w, bool isVignetted);
        Ray(Vec3 _p0, Vec3 _v, double t, double w, bool isVignetted);
        Ray(std::array<double,3> _p0, std::array<double,3> _v,
            double t, double w, bool isVignetted);
        Ray(bool failed);

        Vec3 p0; // reference position
        Vec3 v;  // "velocity" Vec3, really v/c
        double t0; // reference time, really c*t0
        double wavelength; // in vacuum, in nanometers
        bool isVignetted;
        bool failed;

        Vec3 positionAtTime(double t) const;
        bool operator==(const Ray&) const;
        bool operator!=(const Ray&) const;
        double getX0() const { return p0.x; }
        double getY0() const { return p0.y; }
        double getZ0() const { return p0.z; }
        double getVx() const { return v.x; }
        double getVy() const { return v.y; }
        double getVz() const { return v.z; }

        void setFail() { failed=true; }
        void clearFail() { failed=false; }

        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Ray& r) {
        return os << r.repr();
    }
}

#endif
