#include "quadric.h"
#include "utils.h"

namespace batoid {
    Quadric::Quadric(double R, double conic) : _R(R), _conic(conic),
        _Rsq(R*R), _Rinvsq(1./R/R),
        _cp1(conic+1), _cp1inv(1./_cp1),
        _Rcp1(R/_cp1), _RRcp1cp1(R*R/_cp1/_cp1),
        _cp1RR(_cp1/R/R) {}

    double Quadric::sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(_R*(1.+std::sqrt(1.-r2*_cp1RR)));
        return 0.0;
        // Following also works, except leads to divide by 0 when _conic=-1
        // return R/(1+_conic)*(1-std::sqrt(1-(1+_conic)*r2/R/R))+B;
    }

    Vector3d Quadric::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0)
            return Vector3d(0,0,1);
        double dzdr1 = dzdr(r);
        return Vector3d(-dzdr1*x/r, -dzdr1*y/r, 1).normalized();
    }

    bool Quadric::timeToIntersect(const Ray& r, double& t) const {
        double vr2 = r.v[0]*r.v[0] + r.v[1]*r.v[1];
        double vz2 = r.v[2]*r.v[2];
        double vrr0 = r.v[0]*r.p0[0] + r.v[1]*r.p0[1];
        double r02 = r.p0[0]*r.p0[0] + r.p0[1]*r.p0[1];
        double z0term = r.p0[2]-_Rcp1;

        // Quadratic equation coefficients
        double a = vz2 + vr2*_cp1inv;
        double b = 2*(r.v[2]*z0term + vrr0*_cp1inv);
        double c = z0term*z0term - _RRcp1cp1 + r02*_cp1inv;

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
        Vector3d point = r.positionAtTime(t);
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

    bool Quadric::operator==(const Surface& rhs) const {
        if (const Quadric* other = dynamic_cast<const Quadric*>(&rhs)) {
            return _R == other->_R &&
            _conic == other->_conic;
        } else return false;
    }

    std::string Quadric::repr() const {
        std::ostringstream oss(" ");
        oss << "Quadric(" << _R << ", " << _conic << ")";
        return oss.str();
    }

    double Quadric::dzdr(double r) const {
        if (_R != 0.0)
            return r/(_R*std::sqrt(1-r*r*_cp1RR));
        return 0.0;
    }

}
