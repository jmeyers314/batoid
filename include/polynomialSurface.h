#ifndef batoid_PolynomialSurface_h
#define batoid_PolynomialSurface_h

#include <vector>
#include <mutex>
#include "surface.h"
#include "ray.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

namespace batoid {

    class PolynomialSurface : public Surface {
    public:
        PolynomialSurface(MatrixXd coefs);

        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        bool timeToIntersect(const Ray& r, double & t) const override;

        MatrixXd getCoefs() const {return _coefs; }

        PolynomialSurface getGradX() const;
        PolynomialSurface getGradY() const;

    private:
        const MatrixXd _coefs;
        mutable MatrixXd _coefs_gradx;
        mutable MatrixXd _coefs_grady;
        mutable bool _grad_ready{false};
        mutable std::mutex _mtx;

        void computeGradCoefs() const;
    };

    namespace poly {
        double horner2d(double x, double y, const MatrixXd&);
        MatrixXd gradx(const MatrixXd& coefs);
        MatrixXd grady(const MatrixXd& coefs);
    }
}

#endif // batoid_PolynomialSurface_h
