#include "ray.h"

namespace jtrace {
    Ray::Ray(double x0, double y0, double z0, double vx, double vy, double vz, double t=0.0,
             double w=0.0, bool isV=false) :
        p0(Vec3(x0, y0, z0)), v(Vec3(vx, vy, vz).UnitVec3()), t0(t),
        wavelength(w), isVignetted(isV) {}

    Ray::Ray(Vec3 _p0, Vec3 _v, double t=0.0, double w=0.0, bool isV=false) :
        p0(_p0), v(_v.UnitVec3()), t0(t), wavelength(w), isVignetted(isV) {}

    Ray::Ray(std::array<double,3> _p0, std::array<double,3> _v,
             double t=0.0, double w=0.0, bool isV=false) :
        p0(Vec3(_p0)), v(Vec3(_v).UnitVec3()), t0(t), wavelength(w), isVignetted(isV) {}

    std::string Ray::repr() const {
        std::ostringstream oss(" ");
        oss << "Ray(" << p0 << ", " << v;
        if (t0 != 0.0) oss << ", t0=" << t0;
        if (wavelength != 0.0) oss << ", wavelength=" << wavelength;
        if (isVignetted) oss << ", isVignetted=True";
        oss << ")";
        return oss.str();
    }

    Vec3 Ray::operator()(const double t) const {
        return p0+v*(t-t0);
    }

}
