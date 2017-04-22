#include "intersection.h"
#include "utils.h"

namespace jtrace {
    Intersection::Intersection(const double _t, const Vec3 _point, const Vec3 _surfaceNormal) :
        t(_t), point(_point), surfaceNormal(_surfaceNormal.UnitVec3()) {}

    Ray Intersection::reflectedRay(const Ray& r) const {
        double n = 1.0 / r.v.Magnitude();
        Vec3 nv = r.v * n;
        double c1 = DotProduct(nv, surfaceNormal);
        return Ray(point, (nv - 2*c1*surfaceNormal).UnitVec3()/n, t, r.wavelength, r.isVignetted);
    }

    Ray Intersection::refractedRay(const Ray& r, double n1, double n2) const {
        //assert n1 == 1./r.Magnitude()
        Vec3 nv = r.v * n1;
        double alpha = DotProduct(nv, surfaceNormal);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vec3 f1 = (nv+k1*surfaceNormal).UnitVec3();
        Vec3 f2 = (nv+k2*surfaceNormal).UnitVec3();
        if (DotProduct(f1, nv) > DotProduct(f2, nv))
            return Ray(point, f1/n2, t, r.wavelength, r.isVignetted);
        else
            return Ray(point, f2/n2, t, r.wavelength, r.isVignetted);
    }

    Ray Intersection::refractedRay(const Ray& r, const Medium& m1, const Medium& m2) const {
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        return refractedRay(r, n1, n2);
    }

    std::string Intersection::repr() const {
        std::ostringstream oss(" ");
        oss << "Intersection(" << t << ", " << point << ", " << surfaceNormal << ")";
        return oss.str();
    }
}
