#include "bicubic.h"
#include "solve.h"

using Eigen::Vector2d;
using Eigen::Vector4d;
using Eigen::Matrix4d;

namespace batoid {
    // Look up an index.  Use STL binary search.
    int EqArgVec::upperIndex(const double a) const
    {
        if (a<vec.front()-slop || a>vec.back()+slop)
            throw TableOutOfRange(a,vec.front(),vec.back());
        // check for slop
        if (a < vec.front()) return 1;
        if (a > vec.back()) return vec.size()-1;

        int i = int( std::ceil( (a-vec.front()) / da) );
        if (i >= int(vec.size())) --i; // in case of rounding error
        if (i == 0) ++i;
        // check if we need to move ahead or back one step due to rounding errors
        while (a > vec[i]) ++i;
        while (a < vec[i-1]) --i;
        return i;
    }

    // http://www.paulinternet.nl/?page=bicubic
    // p defined at xp=-1, 0, 1, 2
    // Want to interpolate x between 0 and 1
    // This particular implementation yields a Catmull-Rom spline, with
    // a continuous first derivative, but no such guarantee for the second
    // derivative.
    double cubicInterpolate(Vector4d p, double x) {
        return p(1) + 0.5 * x*(p(2) - p(0) + x*(2.0*p(0) - 5.0*p(1) + 4.0*p(2) - p(3) + x*(3.0*(p(1) - p(2)) + p(3) - p(0))));
    }

    double bicubicInterpolate(Matrix4d p, double x, double y) {
        Vector4d arr;
        for (int i=0; i<4; i++) {
            arr(i) = cubicInterpolate(p.row(i), x);
        }
        return cubicInterpolate(arr, y);
    }

    double cubicGradient(Vector4d p, double x) {
        return 0.5 * (p(2) - p(0) + 2*x*(2.0*p(0) - 5.0*p(1) + 4.0*p(2) - p(3) + 3*x*(3.0*(p(1) - p(2)) + p(3) - p(0))));
    }

    Vector2d bicubicGradient(Matrix4d p, double x, double y) {
        Vector4d arr;
        int i=0;
        for (i=0; i<4; i++) {
            arr(i) = cubicInterpolate(p.row(i), x);
        }
        double dzdy = cubicGradient(arr, y);

        for (i=0; i<4; i++) {
            arr(i) = cubicInterpolate(p.col(i), y);
        }
        double dzdx = cubicGradient(arr, x);

        return Vector2d(dzdx, dzdy);
    }

    Bicubic::Bicubic(const std::vector<double> xs, const std::vector<double> ys, const MatrixXd zs) :
        _xargs(xs), _yargs(ys), _zs(zs) {}

    double Bicubic::sag(double x, double y) const {
        // Determine cell to use
        int ix = _xargs.upperIndex(x);
        int iy = _yargs.upperIndex(y);
        MatrixXd block = _zs.block<4,4>(iy-2, ix-2);
        // Do single cell interpolation
        return bicubicInterpolate(
            block,
            (x-_xargs[ix-1])/_xargs.getDa(),
            (y-_yargs[iy-1])/_yargs.getDa()
        );
    }

    Vector3d Bicubic::normal(double x, double y) const {
        // Determine cell to use
        int ix = _xargs.upperIndex(x);
        int iy = _yargs.upperIndex(y);
        MatrixXd block = _zs.block<4,4>(iy-2, ix-2);
        // Do single cell interpolation
        Vector2d dzdxy = bicubicGradient(
            block,
            (x-_xargs[ix-1])/_xargs.getDa(),
            (y-_yargs[iy-1])/_yargs.getDa()
        );
        return Vector3d(-dzdxy(0)/_xargs.getDa(), -dzdxy(1)/_yargs.getDa(), 1).normalized();
    }

    class BicubicResidual {
    public:
        BicubicResidual(const Bicubic& b, const Ray& r) : _b(b), _r(r) {}
        double operator()(double t) const {
            Vector3d p = _r.positionAtTime(t);
            return _b.sag(p(0), p(1)) - p(2);
        }
    private:
        const Bicubic& _b;
        const Ray& _r;
    };

    bool Bicubic::timeToIntersect(const Ray& r, double& t) const {
        // Guess that 0.0 is a good inital estimate
        t = 0.0;
        BicubicResidual resid(*this, r);
        Solve<BicubicResidual> solve(resid, t, t+1e-2);
        solve.setMethod(Method::Brent);
        solve.setXTolerance(1e-12);

        try {
            solve.bracket();
            t = solve.root();
        } catch (const SolveError&) {
            return false;
        }
        if (t < r.t) return false;
        return true;
    }

}
