#include "plane.h"
#include "utils.h"
#include <cmath>

namespace batoid {
    bool Plane::timeToIntersect(const Ray& r, double& t) const {
        t = -r.r[2]/r.v[2] + r.t;
        return (_allowReverse || t >= r.t);
    }
}
