#ifndef __jem_ray__h
#define __jem_ray__h

#include <sstream>
#include <string>
#include "vec3.h"

namespace jtrace {
    struct Ray {
        Ray(double x0, double y0, double z0, double vx, double vy, double vz, double t);
        Ray(Vec3 _p0, Vec3 _v, double t);
        Vec3 p0; // reference position
        Vec3 v;  // "velocity" Vec3
        double t0; // reference time
        Vec3 operator()(double t) const;
        double getX0() const { return p0.x; }
        double getY0() const { return p0.y; }
        double getZ0() const { return p0.z; }
        double getVx() const { return v.x; }
        double getVy() const { return v.y; }
        double getVz() const { return v.z; }

        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Ray& r);
}

#endif
