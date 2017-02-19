#include "asphere.h"
#include "solve.h"
#include "quadric.h"

namespace jtrace {
    Asphere::Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B) :
        R(_R), kappa(_kappa), alpha(_alpha), B(_B) {}

    double Asphere::operator()(double x, double y) const {
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
        double dzdr1 = dzdr(r);
        Vec3 n{-dzdr1*x/r, -dzdr1*y/r, 1};
        return n.UnitVec3();
    }

    class AsphereResidual {
    public:
        AsphereResidual(const Asphere& _a, const Ray& _r) : a(_a), r(_r) {}
        double operator()(double t) const {
            Vec3 p = r(t);
            double resid = a(p.x, p.y) - p.z;
            return resid;
        }
    private:
        const Asphere& a;
        const Ray& r;
    };

    Intersection Asphere::intersect(const Ray& r) const {
        // Solve the quadric problem analytically to get a good starting point.
        Quadric quad(R, kappa, B);
        Intersection isec = quad.intersect(r);

        AsphereResidual resid(*this, r);
        Solve<AsphereResidual> solve(resid, isec.t, isec.t+1e-2);
        solve.setMethod(Method::Brent);
        solve.setXTolerance(1e-12);
        solve.bracket();
        double t = solve.root();

        Vec3 point = r(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
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
