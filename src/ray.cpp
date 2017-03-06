#include "ray.h"

namespace jtrace {
    Ray::Ray(double x0, double y0, double z0, double vx, double vy, double vz, double t=0.0) :
        p0(Vec3(x0, y0, z0)), v(Vec3(vx, vy, vz).UnitVec3()), t0(t) {}

    Ray::Ray(Vec3 _p0, Vec3 _v, double t) : p0(_p0), v(_v.UnitVec3()), t0(t) {}

    Ray::Ray(std::array<double,3> _p0, std::array<double,3> _v, double t) :
        p0(Vec3(_p0)), v(Vec3(_v).UnitVec3()), t0(t) {}

    std::string Ray::repr() const {
        std::ostringstream oss(" ");
        oss << "Ray(" << p0 << ", " << v << ", " << t0 << ")";
        return oss.str();
    }

    Vec3 Ray::operator()(const double t) const {
        return p0+v*(t-t0);
    }

}
