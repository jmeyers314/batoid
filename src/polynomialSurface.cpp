#include "polynomialSurface.h"
#include "solve.h"

namespace batoid {
    PolynomialSurface::PolynomialSurface(MatrixXd coefs) : _coefs(coefs) {}

    double PolynomialSurface::sag(double x, double y) const {
        return poly::horner2d(x, y, _coefs);
    }

    Vector3d PolynomialSurface::normal(double x, double y) const {
        if (!_grad_ready)
            computeGradCoefs();
        double dzdx = poly::horner2d(x, y, _coefs_gradx);
        double dzdy = poly::horner2d(x, y, _coefs_grady);
        return Vector3d(-dzdx, -dzdy, 1).normalized();
    }

    class PolynomialSurfaceResidual {
    public:
        PolynomialSurfaceResidual(const PolynomialSurface& ps, const Ray& r) : _ps(ps), _r(r) {}
        double operator()(double t) const {
            Vector3d p = _r.positionAtTime(t);
            return _ps.sag(p(0), p(1)) - p(2);
        }
    private:
        const PolynomialSurface& _ps;
        const Ray& _r;
    };

    bool PolynomialSurface::timeToIntersect(const Ray& r, double& t) const {
        // Guess that 0.0 is a good inital estimate
        t = 0.0;
        PolynomialSurfaceResidual resid(*this, r);
        Solve<PolynomialSurfaceResidual> solve(resid, t, t+1e-2);
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

    void PolynomialSurface::computeGradCoefs() const {
        std::lock_guard<std::mutex> lock(_mtx);
        if (!_grad_ready) {
            _coefs_gradx = poly::gradx(_coefs);
            _coefs_grady = poly::grady(_coefs);
        }
    }

    namespace poly {
        double horner(double x, const VectorXd& coefs) {
            double result = 0;
            for (int i=coefs.size()-1; i>=0; i--) {
                result *= x;
                result += coefs(i);
            }
            return result;
        }

        double horner2d(double x, double y, const MatrixXd& coefs) {
            double result = 0.0;
            for (int i=coefs.rows()-1; i>=0; i--) {
                result *= x;
                result += horner(y, coefs.row(i));
            }
            return result;
        }

        MatrixXd gradx(const MatrixXd& coefs) {
            MatrixXd result = MatrixXd::Zero(coefs.rows()-1, coefs.cols());
            for(int i=1; i<coefs.rows(); i++) {
                for(int j=0; j<coefs.cols(); j++) {
                    if (coefs(i, j) == 0.0) continue;
                    result(i-1, j) = coefs(i, j)*i;
                }
            }
            return result;
        }

        MatrixXd grady(const MatrixXd& coefs) {
            MatrixXd result = MatrixXd::Zero(coefs.rows(), coefs.cols()-1);
            for(int i=0; i<coefs.rows(); i++) {
                for(int j=1; j<coefs.cols(); j++) {
                    if (coefs(i, j) == 0.0) continue;
                    result(i, j-1) = coefs(i, j)*j;
                }
            }
            return result;
        }


    }

}
