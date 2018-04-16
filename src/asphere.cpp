#include "asphere.h"
#include "quadric.h"
#include "solve.h"
#include <cmath>

namespace batoid {
    Asphere::Asphere(double R, double conic, std::vector<double> coefs) :
        Quadric(R, conic), _coefs(coefs) {}

    double Asphere::sag(double x, double y) const {
        double r2 = x*x + y*y;
        double rr = r2;
        double result = Quadric::sag(x, y);
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

    bool Asphere::timeToIntersect(const Ray& r, double& t) const {
        // Solve the quadric problem analytically to get a good starting point.
        if (!Quadric::timeToIntersect(r, t))
            return false;

        AsphereResidual resid(*this, r);
        Solve<AsphereResidual> solve(resid, t, t+1e-2);
        solve.setMethod(Method::Brent);
        solve.setXTolerance(1e-12);

        try {
            solve.bracket();
            t = solve.root();
        } catch (const SolveError&) {
            return false;
        }
        return true;
    }

    Ray Asphere::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Asphere::intersectInPlace(Ray& r) const {
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

    std::string Asphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Asphere(" << getR() << ", " << getConic() << ", [";
        if (_coefs.size() == 0) {
            oss << "])";
            return oss.str();
        }
        size_t i=0;
        for (; i<_coefs.size()-1; i++)
            oss << _coefs[i] << ", ";
        oss << _coefs[i] << "])";
        return oss.str();
    }

    double Asphere::dzdr(double r) const {
        double result = Quadric::dzdr(r);
        double rrr = r*r*r;
        int twoi=0;
        for (const auto& c : _coefs) {
            result += (4+twoi)*c*rrr;
            rrr *= r*r;
            twoi += 2;
        }
        return result;
    }

}
