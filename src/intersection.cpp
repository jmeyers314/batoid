#include "intersection.h"

namespace jtrace {

    Intersection::Intersection(const double _t, const Vec3 _point,
                               const Vec3 _surfaceNormal, const Surface* _surface) :
        t(_t), point(_point), surfaceNormal(_surfaceNormal.UnitVec3()), surface(_surface) {}

    Ray Intersection::reflectedRay(const Ray &r) const {
        double c1 = DotProduct(r.v, surfaceNormal);
        return Ray(point, (r.v - 2*c1*surfaceNormal).UnitVec3(), t);
    }

    std::string Intersection::repr() const {
        std::ostringstream oss(" ");
        oss << "Intersection(" << t << ", " << point << ", " << surfaceNormal << ")";
        return oss.str();
    }
}
