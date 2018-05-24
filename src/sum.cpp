#include "sum.h"
#include "solve.h"


namespace batoid {

    // Note not commutative for now.
    Sum::Sum(std::vector<std::shared_ptr<Surface>> surfaces) :
        _surfaces(surfaces) {}

    double Sum::sag(double x, double y) const {
        double result = 0.0;
        for (const auto& s : _surfaces)
            result += s->sag(x, y);
        return result;
    }

    Vector3d Sum::normal(double x, double y) const {
        Vector3d result{0,0,0};
        for (const auto& s : _surfaces) {
            Vector3d norm{s->normal(x, y)};
            norm /= norm[2];  //normalize to unit z-component
            result += norm;
        }
        return result.normalized();
    }

    Ray Sum::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Sum::intersectInPlace(Ray& r) const {
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

    class SumResidual {
    public:
        SumResidual(const Sum& s, const Ray& r) : _s(s), _r(r) {}
        double operator()(double t) const {
            Vector3d p = _r.positionAtTime(t);
            return _s.sag(p(0), p(1)) - p(2);
        }
    private:
        const Sum& _s;
        const Ray& _r;
    };

    bool Sum::timeToIntersect(const Ray& r, double& t) const {
        // better guess?
        t = 0.0;
        SumResidual resid(*this, r);
        Solve<SumResidual> solve(resid, t, t+1e-2);
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

    std::string Sum::repr() const {
        return std::string("Sum");
    }
}
