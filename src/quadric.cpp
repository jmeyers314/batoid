#include "quadric.h"
#include "utils.h"
#include "except.h"

namespace batoid {
    Quadric::Quadric(double R, double conic) : _R(R), _conic(conic) {}

    double Quadric::sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(_R*(1.+std::sqrt(1.-(1.+_conic)*r2/_R/_R)));
        return 0.0;
        // Following also works, except leads to divide by 0 when _conic=-1
        // return R/(1+_conic)*(1-std::sqrt(1-(1+_conic)*r2/R/R))+B;
    }

    Vec3 Quadric::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0) return Vec3(0,0,1);
        double dzdr1 = dzdr(r);
        return Vec3(-dzdr1*x/r, -dzdr1*y/r, 1).UnitVec3();
    }

    bool Quadric::timeToIntersect(const Ray& r, double& t) const {
        double vr2 = r.v.x*r.v.x + r.v.y*r.v.y;
        double vz2 = r.v.z*r.v.z;
        double vrr0 = r.v.x*r.p0.x + r.v.y*r.p0.y;
        double r02 = r.p0.x*r.p0.x + r.p0.y*r.p0.y;
        double z0term = (r.p0.z-_R/(1+_conic));

        // Quadratic equation coefficients
        double a = vz2 + vr2/(1+_conic);
        double b = 2*r.v.z*z0term + 2*vrr0/(1+_conic);
        double c = z0term*z0term - _R*_R/(1+_conic)/(1+_conic) + r02/(1+_conic);

        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        // Should probably check the solutions here since we obtained the quadratic
        // formula above by squaring both sides of an equation.

        if (n == 0)
            return false;
        else if (n == 1) {
            if (r1 < 0)
                return false;
            t = r1;
        } else {
            if (r1 < 0) {
                if (r2 < 0)
                    return false;
                else
                    t = r2;
            } else
                t = std::min(r1, r2);
        }
        t += r.t0;

        return true;
    }

    Ray Quadric::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Quadric::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntersect(r, t)) {
            r.failed=true;
            return;
        }
        r.p0 = r.positionAtTime(t);
        r.t0 = t;
        return;
    }

    std::string Quadric::repr() const {
        std::ostringstream oss(" ");
        oss << "Quadric(" << _R << ", " << _conic << ")";
        return oss.str();
    }

    double Quadric::dzdr(double r) const {
        if (_R != 0.0)
            return r/(_R*std::sqrt(1-r*r*(1+_conic)/_R/_R));
        return 0.0;
    }

}
