#include "asphere.h"
#include "quadric.h"
#include "solve.h"
#include <cmath>

namespace batoid {
    Asphere::Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B,
                     double _Rin, double _Rout) :
        R(_R), kappa(_kappa), alpha(_alpha), B(_B), Rin(_Rin), Rout(_Rout) {}

    double Asphere::sag(double x, double y) const {
        double r2 = x*x + y*y;
        double result = B;
        if (R != 0) {
            double den = R*(1.+std::sqrt(1.-(1.+kappa)*r2/R/R));
            result += r2/den;
        }
        double rr = r2;
        for (const auto& a : alpha) {
            rr *= r2;
            result += rr*a;
        }
        return result;
    }

    Vec3 Asphere::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0) return Vec3(0,0,1);
        double dzdr1 = dzdr(r);
        Vec3 n{-dzdr1*x/r, -dzdr1*y/r, 1};
        return n.UnitVec3();
    }

    class AsphereResidual {
    public:
        AsphereResidual(const Asphere& _a, const Ray& _r) : a(_a), r(_r) {}
        double operator()(double t) const {
            Vec3 p = r.positionAtTime(t);
            double resid = a.sag(p.x, p.y) - p.z;
            return resid;
        }
    private:
        const Asphere& a;
        const Ray& r;
    };

    Ray Asphere::intercept(const Ray& r) const {
        if (r.failed)
            return Ray(true);
        // Solve the quadric problem analytically to get a good starting point.
        Quadric quad(R, kappa, B);
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
        Quadric quad(R, kappa, B);
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
        double rho = std::hypot(point.x, point.y);
        bool isVignetted = rho < Rin || rho > Rout;
        return Intersection(t, point, surfaceNormal, isVignetted);
    }

    std::string Asphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Asphere(" << R << ", " << kappa << ", [";
        for (const auto& a : alpha) {
            oss << a << ", ";
        }
        oss << "], " << B << ")";
        return oss.str();
    }

    double Asphere::dzdr(double r) const {
        double result = r/(R*std::sqrt(1-r*r*(1+kappa)/R/R));
        double rrr = r*r*r;
        int twoi=0;
        for (const auto& a : alpha) {
            result += (4+twoi)*a*rrr;
            rrr *= r*r;
            twoi += 2;
        }
        return result;
    }

    inline std::ostream& operator<<(std::ostream& os, const Asphere& a) {
        return os << a.repr();
    }

}
