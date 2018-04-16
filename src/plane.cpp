#include "plane.h"
#include <cmath>

namespace batoid {
    Ray Plane::intercept(const Ray& r) const {
        if (r.failed) return r;
        double t = -r.p0.z/r.v.z + r.t0;
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Plane::interceptInPlace(Ray& r) const {
        if (r.failed) return;
        double t = -r.p0.z/r.v.z + r.t0;
        r.p0 = r.positionAtTime(t);
        r.t0 = t;
    }

    std::string Plane::repr() const {
        return std::string("Plane()");
    }
}
