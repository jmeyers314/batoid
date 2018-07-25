#include "plane.h"
#include <cmath>

namespace batoid {
    bool Plane::timeToIntersect(const Ray& r, double& t) const {
        t = -r.r[2]/r.v[2] + r.t;
        if (t < r.t) return false;
        return true;
    }

    bool Plane::operator==(const Surface& rhs) const {
        return bool(dynamic_cast<const Plane*>(&rhs));
    }
}
