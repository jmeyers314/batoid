#ifndef batoid_zernike_h
#define batoid_zernike_h

#include <vector>
#include <mutex>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace batoid {

    class Zernike : public Surface {
    public:
        Zernike(std::vector<double> coefs, double R_outer=1.0, double R_inner=0.0);

        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        const std::vector<double>& getCoefs() const { return _coefs; }
        double getROuter() const { return _R_outer; }
        double getRInner() const { return _R_inner; }
        std::string repr() const;

    private:
        const std::vector<double> _coefs;
        const double _R_outer, _R_inner;
        mutable MatrixXd _coef_array;
        mutable MatrixXd _coef_array_gradx;
        mutable MatrixXd _coef_array_grady;
        mutable bool _coef_array_ready{false};
        mutable bool _coef_array_grad_ready{false};
        mutable std::mutex _mtx;

        bool timeToIntersect(const Ray& r, double& t) const;
    };

    namespace zernike {
        unsigned long long nCr(unsigned n, unsigned k);
        std::vector<double> binomial(double a, double b, int n);
        double horner2d(double x, double y, const MatrixXd&);

        std::pair<int,int> noll_to_zern(int j);
        double zern_norm(int n, int m);
        std::vector<double> zern_rho_coefs(int n, int m);
        double h(int m, int j, double eps);
        std::vector<double> Q(int m, int j, double eps);
        double Q0(int m, int j, double eps);
        std::vector<double> annular_zern_rho_coefs(int n, int m, double eps);

        MatrixXcd xy_contribution(int rho2_power, int rho_power, std::pair<int,int> shape);
        MatrixXd rrsq_to_xy(const MatrixXcd& coefs, std::pair<int,int> shape);

        MatrixXcd zern_coef_array(int n, int m, double eps, std::pair<int,int> shape);
        std::vector<MatrixXcd> noll_coef_array(int jmax, double eps);
        std::vector<MatrixXd> noll_coef_array_xy(int jmax, double eps);
    }
}

#endif
