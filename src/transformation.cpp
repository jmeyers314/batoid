#include "transformation.h"
#include "except.h"

namespace jtrace {
    Transformation::Transformation(std::shared_ptr<const Surface> s, double dx, double dy, double dz) :
        transformee(s), dr(Vec3(dx, dy, dz)), rot(ident_rot) {}

    Transformation::Transformation(std::shared_ptr<const Surface> s, const Vec3& _dr) :
        transformee(s), dr(_dr), rot(ident_rot) {}

    Transformation::Transformation(std::shared_ptr<const Surface> s, const Rot3& r) :
        transformee(s), dr(), rot(r) {}

    double Transformation::operator()(double x, double y) const {
        throw NotImplemented("Transformation::operator() not implemented");
    }

    Vec3 Transformation::normal(double x, double y) const {
        throw NotImplemented("Transformation::normal() not implemented");
    }

    Intersection Transformation::intersect(const Ray& r) const {
        if (r.failed)
            return Intersection(true);
        // Need to transform the coord sys of r into the coord sys of the transformee.
        Ray rr {RotVec(rot, r.p0-dr), RotVec(rot, r.v), r.t0, r.wavelength, r.isVignetted};
        Intersection isec = transformee->intersect(rr);
        // Now transform intersection back into transformed coord sys.
        return Intersection(isec.t, UnRotVec(rot, isec.point)+dr, UnRotVec(rot, isec.surfaceNormal), isec.isVignetted);
    }

    std::string Transformation::repr() const {
        std::ostringstream oss(" ");
        oss << "Transformation(" << transformee->repr() << ", " << dr << ")";
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Transformation& t) {
        return os << t.repr();
    }
}
