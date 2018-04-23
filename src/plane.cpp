#include "plane.h"
#include <cmath>

namespace batoid {
    Ray Plane::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t = -r.p0[2]/r.v[2] + r.t0;
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Plane::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t = -r.p0[2]/r.v[2] + r.t0;
        r.p0 = r.positionAtTime(t);
        r.t0 = t;
    }

    std::string Plane::repr() const {
        return std::string("Plane()");
    }
}
