#include "transformation.h"

namespace jtrace {
    Transformation::Transformation(const Surface* s, double dx, double dy, double dz) :
        transformee(s), dr(Vec3(dx, dy, dz)), rot3{ident_rot3} {}

    Transformation::Transformation(const Surface* s, const Vec3& _dr) :
        transformee(s), dr(_dr), rot3{ident_rot3} {}

    Transformation::Transformation(const Surface *s, std::array<std::array<double, 3>, 3> r) :
        transformee(s), dr(), rot3(r) {}

    double Transformation::operator()(double x, double y) const {
        throw NotImplemented("Transformation::operator() not implemented");
    }

    Vec3 Transformation::normal(double x, double y) const {
        throw NotImplemented("Transformation::normal() not implemented");
    }

    Intersection Transformation::intersect(const Ray &r) const {
        // Need to transform the coord sys of r into the coord sys of the transformee.
        Ray rr {r.p0-dr, r.v, r.t0};
        Intersection isec = transformee->intersect(rr);
        // Now transform intersection back into transformed coord sys.
        return Intersection(isec.t, isec.point+dr, isec.surfaceNormal, this);
    }

    std::string Transformation::repr() const {
        std::ostringstream oss(" ");
        oss << "Transformation(" << transformee->repr() << ", " << dr << ")";
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Transformation &t) {
        return os << t.repr();
    }
}
