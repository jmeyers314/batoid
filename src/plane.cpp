#include "plane.h"
#include <cmath>

namespace batoid {
    Ray Plane::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t = -r.r[2]/r.v[2] + r.t;
        if (t < r.t)
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.vignetted);
    }

    void Plane::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t = -r.r[2]/r.v[2] + r.t;
        if (t < r.t) {
            r.failed=true;
            return;
        }
        r.r = r.positionAtTime(t);
        r.t = t;
    }

    std::string Plane::repr() const {
        return std::string("Plane()");
    }

    bool Plane::operator==(const Surface& rhs) const {
        return bool(dynamic_cast<const Plane*>(&rhs));
    }
}
