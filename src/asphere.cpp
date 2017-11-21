#include "asphere.h"
#include "quadric.h"
#include "solve.h"
#include <cmath>

namespace batoid {
    Asphere::Asphere(double R, double conic, std::vector<double> coefs) :
        _R(R), _conic(conic), _coefs(coefs) {}

    double Asphere::sag(double x, double y) const {
        double r2 = x*x + y*y;
        double result = 0.0;
        if (_R != 0) result += r2/(_R*(1.+std::sqrt(1.-(1.+_conic)*r2/_R/_R)));
        double rr = r2;
        for (const auto& c : _coefs) {
            rr *= r2;
            result += c*rr;
        }
        return result;
    }

    Vec3 Asphere::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0) return Vec3(0,0,1);
        double dzdr1 = dzdr(r);
        return Vec3(-dzdr1*x/r, -dzdr1*y/r, 1).UnitVec3();
    }

    class AsphereResidual {
    public:
        AsphereResidual(const Asphere& a, const Ray& r) : _a(a), _r(r) {}
        double operator()(double t) const {
            Vec3 p = _r.positionAtTime(t);
            return _a.sag(p.x, p.y) - p.z;
        }
    private:
        const Asphere& _a;
        const Ray& _r;
    };

    Ray Asphere::intercept(const Ray& r) const {
        if (r.failed)
            return Ray(true);
        // Solve the quadric problem analytically to get a good starting point.
        Quadric quad(_R, _conic);
        Ray rquad = quad.intercept(r);
        if (rquad.failed)
            return Ray(true);

        AsphereResidual resid(*this, r);
        Solve<AsphereResidual> solve(resid, rquad.t0, rquad.t0+1e-2);
        solve.setMethod(Method::Brent);
        solve.setXTolerance(1e-12);
        double t;
        try {
            solve.bracket();
            t = solve.root();
        } catch (const SolveError&) {
            return Ray(true);
        }

        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    Intersection Asphere::intersect(const Ray& r) const {
        if (r.failed)
            return Intersection(true);
        // Solve the quadric problem analytically to get a good starting point.
        Quadric quad(_R, _conic);
        Intersection isec = quad.intersect(r);
        if (isec.failed)
            return isec;

        AsphereResidual resid(*this, r);
        Solve<AsphereResidual> solve(resid, isec.t, isec.t+1e-2);
        solve.setMethod(Method::Brent);
        solve.setXTolerance(1e-12);
        double t;
        try {
            solve.bracket();
            t = solve.root();
        } catch (const SolveError&) {
            return Intersection(true);
        }

        Vec3 point = r.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
    }

    std::string Asphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Asphere(" << _R << ", " << _conic << ", [";
        for (const auto& c : _coefs) {
            oss << c << ", ";
        }
        oss << "])";
        return oss.str();
    }

    double Asphere::dzdr(double r) const {
        double result = 0.0;
        if (_R != 0.0)
            result += r/(_R*std::sqrt(1-r*r*(1+_conic)/_R/_R));
        double rrr = r*r*r;
        int twoi=0;
        for (const auto& c : _coefs) {
            result += (4+twoi)*c*rrr;
            rrr *= r*r;
            twoi += 2;
        }
        return result;
    }

    inline std::ostream& operator<<(std::ostream& os, const Asphere& a) {
        return os << a.repr();
    }

}
