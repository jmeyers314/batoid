#include "asphere.h"
#include "solve.h"
#include "paraboloid.h"

namespace jtrace {
    Asphere::Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B) :
        R(_R), kappa(_kappa), alpha(_alpha), B(_B) {}

    double Asphere::operator()(double x, double y) const {
        double r2 = x*x + y*y;
        double den = R*(1.+std::sqrt(1.-(1.+kappa)*r2/R/R));
        double result = r2/den + B;
        double rr = r2;
        for (const auto &a : alpha) {
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
        AsphereResidual(const Asphere &_a, const Ray &_r) : a(_a), r(_r) {}
        double operator()(double t) const {
            Vec3 p = r(t);
            return a(p.x, p.y) - p.z;
        }
    private:
        const Asphere &a;
        const Ray &r;
    };

    Intersection Asphere::intersect(const Ray &r) const {
        // Solve the paraboloid problem analytically to get a good starting point.
        double A = 1./(2*R);
        Paraboloid para(A, B);
        Intersection isec = para.intersect(r);
        double zpara = isec.point.z;
        double zasphere = (*this)(isec.point.x, isec.point.y);
        Solve<AsphereResidual> solve(AsphereResidual(*this, r), isec.t, isec.t+1e-3);

        double t;
        if (zpara > zasphere) {
            if (r.v.z < 0) {
                solve.bracketUpper();
                t = solve.root();
            } else {
                solve.bracketLower();
                t = solve.root();
            }
        } else {
            if (r.v.z < 0) {
                solve.bracketLower();
                t = solve.root();
            } else {
                solve.bracketUpper();
                t = solve.root();
            }
        }

        Vec3 point = r(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal, this);
    }

    std::string Asphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Asphere(" << R << ", " << kappa << ", [";
        for (const auto &a : alpha) {
            oss << a << ", ";
        }
        oss << "], " << B << ")";
        return oss.str();
    }

    double Asphere::dzdr(double r) const {
        double result = r/(R*std::sqrt(1-r*r*(1+kappa)/R/R));
        double rrr = r*r*r;
        int twoi=0;
        for (const auto &a : alpha) {
            result += (4+twoi)*a*rrr;
            rrr *= r*r;
            twoi += 2;
        }
        return result;
    }

    inline std::ostream& operator<<(std::ostream& os, const Asphere &a) {
        return os << a.repr();
    }

}
