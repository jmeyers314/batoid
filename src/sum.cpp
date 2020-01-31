#include "sum.h"
#include "solve.h"


namespace batoid {

    Sum::Sum(const std::vector<std::shared_ptr<Surface>> surfaces) :
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
        result[2] = 1; // reset z-component
        return result.normalized();
    }

    // class SumResidual {
    // public:
    //     SumResidual(const Sum& s, const Ray& r) : _s(s), _r(r) {}
    //     double operator()(double t) const {
    //         Vector3d p = _r.positionAtTime(t);
    //         return _s.sag(p(0), p(1)) - p(2);
    //     }
    // private:
    //     const Sum& _s;
    //     const Ray& _r;
    // };
    //
    // bool Sum::timeToIntersect(const Ray& r, double& t) const {
    //     // Use first surface as an initial guess
    //     if (!_surfaces[0]->timeToIntersect(r, t))
    //         return false;
    //     SumResidual resid(*this, r);
    //     Solve<SumResidual> solve(resid, t, t+1e-2);
    //     solve.setMethod(Method::Brent);
    //     solve.setXTolerance(1e-12);
    //
    //     try {
    //         solve.bracket();
    //         t = solve.root();
    //     } catch (const SolveError&) {
    //         return false;
    //     }
    //     if (t < r.t) return false;
    //     return true;
    // }

    bool Sum::timeToIntersect(const Ray& r, double& t) const {
        // Use first surface as an initial guess
        if (!_surfaces[0]->timeToIntersect(r, t))
            return false;
        bool success = Surface::timeToIntersect(r, t);
        return (success && t >= r.t);
    }
}
