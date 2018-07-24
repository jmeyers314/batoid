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
        Zernike(Zernike&& z);

        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        virtual bool operator==(const Surface&) const override;
        bool timeToIntersect(const Ray& r, double& t) const override;

        const std::vector<double>& getCoefs() const { return _coefs; }
        double getROuter() const { return _R_outer; }
        double getRInner() const { return _R_inner; }
        std::string repr() const override;

        Zernike getGradX() const;
        Zernike getGradY() const;

    private:
        const std::vector<double> _coefs;
        const double _R_outer, _R_inner;
        mutable MatrixXd _coef_array;
        mutable std::vector<double> _coefs_gradx;
        mutable std::vector<double> _coefs_grady;
        mutable MatrixXd _coefx_array;
        mutable MatrixXd _coefy_array;
        mutable bool _coef_array_ready{false};
        mutable bool _coef_array_grad_ready{false};
        mutable std::mutex _mtx;

        void computeCoefArray() const;
        void computeGradCoefs() const;
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
        std::vector<MatrixXd> noll_coef_array_xy_gradx(int jmax, double eps);
        std::vector<MatrixXd> noll_coef_array_xy_grady(int jmax, double eps);
        MatrixXd noll_coef_array_gradx(int jmax, double eps);
        MatrixXd noll_coef_array_grady(int jmax, double eps);
    }
}

#endif
