#include "bicubic.h"
#include "solve.h"
#include "table.h"
#include <cmath>
#include <limits>

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

    Bicubic::Bicubic(
        const std::vector<double> xs, const std::vector<double> ys, const DRef<MatrixXd> zs,
        const DRef<MatrixXd> dzdxs, const DRef<MatrixXd> dzdys, const DRef<MatrixXd> d2zdxdys,
        const double slopFrac
    ) :     _xargs(xs, slopFrac), _yargs(ys, slopFrac), _zs(zs),
            _dzdxs(dzdxs), _dzdys(dzdys), _d2zdxdys(d2zdxdys)
    {}

    double Bicubic::oneDSpline(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        double d = val0;

        return d + x*(c + x*(b + x*a));
    }

    double Bicubic::oneDGrad(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        return c + x*(2*b + x*3*a);
    }

    double Bicubic::sag(double x, double y) const {
        // Determine cell to use and offset
        int ix, iy;
        try {
            ix = _xargs.upperIndex(x);
            iy = _yargs.upperIndex(y);
        } catch (const TableOutOfRange&) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double dx = _xargs.getDa();
        double dy = _yargs.getDa();
        double xfrac = (x-_xargs[ix-1])/dx;
        double yfrac = (y-_yargs[iy-1])/dy;

        // Interpolate
        double val0 = oneDSpline(xfrac, _zs(iy-1, ix-1), _zs(iy-1, ix),
                                 _dzdxs(iy-1, ix-1)*dx, _dzdxs(iy-1, ix)*dx);

        double val1 = oneDSpline(xfrac, _zs(iy, ix-1), _zs(iy, ix),
                                 _dzdxs(iy, ix-1)*dx, _dzdxs(iy, ix)*dx);

        double der0 = oneDSpline(xfrac, _dzdys(iy-1, ix-1), _dzdys(iy-1, ix),
                                 _d2zdxdys(iy-1, ix-1)*dx, _d2zdxdys(iy-1, ix)*dx);

        double der1 = oneDSpline(xfrac, _dzdys(iy, ix-1), _dzdys(iy, ix),
                                 _d2zdxdys(iy, ix-1)*dx, _d2zdxdys(iy, ix)*dx);

        return oneDSpline(yfrac, val0, val1, der0*dy, der1*dy);
    }

    Vector3d Bicubic::normal(double x, double y) const {
        // Determine cell to use and offset
        int ix, iy;
        try {
            ix = _xargs.upperIndex(x);
            iy = _yargs.upperIndex(y);
        } catch (const TableOutOfRange&) {
            return Vector3d(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()
            );
        }
        double dx = _xargs.getDa();
        double dy = _yargs.getDa();
        double xfrac = (x-_xargs[ix-1])/dx;
        double yfrac = (y-_yargs[iy-1])/dy;

        // x-gradient
        double val0 = oneDGrad(xfrac, _zs(iy-1, ix-1), _zs(iy-1, ix),
                                      _dzdxs(iy-1, ix-1)*dx, _dzdxs(iy-1, ix)*dx);
        double val1 = oneDGrad(xfrac, _zs(iy, ix-1), _zs(iy, ix),
                                      _dzdxs(iy, ix-1)*dx, _dzdxs(iy, ix)*dx);
        double der0 = oneDGrad(xfrac, _dzdys(iy-1, ix-1), _dzdys(iy-1, ix),
                                      _d2zdxdys(iy-1, ix-1)*dx, _d2zdxdys(iy-1, ix)*dx);
        double der1 = oneDGrad(xfrac, _dzdys(iy, ix-1), _dzdys(iy, ix),
                                      _d2zdxdys(iy, ix-1)*dx, _d2zdxdys(iy, ix)*dx);
        double gradx = oneDSpline(yfrac, val0, val1, der0*dy, der1*dy)/dx;

        // y-gradient
        val0 = oneDGrad(yfrac, _zs(iy-1, ix-1), _zs(iy, ix-1),
                               _dzdys(iy-1, ix-1)*dy, _dzdys(iy, ix-1)*dy);
        val1 = oneDGrad(yfrac, _zs(iy-1, ix), _zs(iy, ix),
                               _dzdys(iy-1, ix)*dy, _dzdys(iy, ix)*dy);
        der0 = oneDGrad(yfrac, _dzdxs(iy-1, ix-1), _dzdxs(iy, ix-1),
                               _d2zdxdys(iy-1, ix-1)*dy, _d2zdxdys(iy, ix-1)*dy);
        der1 = oneDGrad(yfrac, _dzdxs(iy-1, ix), _dzdxs(iy, ix),
                               _d2zdxdys(iy-1, ix)*dy, _d2zdxdys(iy, ix)*dy);
        double grady = oneDSpline(xfrac, val0, val1, der0*dx, der1*dx)/dy;
        return Vector3d(-gradx, -grady, 1).normalized();
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
        solve.setXTolerance(1e-14);

        try {
            solve.bracket();
            t = solve.root();
        } catch (const SolveError&) {
            return false;
        } catch (const TableOutOfRange&) {
            return false;
        }
        if (t < r.t) return false;
        return true;
    }

}
