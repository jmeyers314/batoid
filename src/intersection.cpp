#include "intersection.h"

namespace jtrace {

    Intersection::Intersection(const double _t, const Vec3 _point,
                               const Vec3 _surfaceNormal, const Surface* _surface) :
        t(_t), point(_point), surfaceNormal(_surfaceNormal), surface(_surface) {}

    std::string Intersection::repr() const {
        std::ostringstream oss(" ");
        oss << "Intersection(" << t << ", " << point << ", " << surfaceNormal << ")";
        return oss.str();
    }
}
