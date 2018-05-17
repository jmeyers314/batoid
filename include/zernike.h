#ifndef batoid_zernike_h
#define batoid_zernike_h

#include <vector>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Vector3d;

namespace batoid {

    class Zernike : public Surface {
    public:
        Zernike(std::vector<double> coefs, double R_outer=1.0, double R_inner=0.0);

        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        const std::vector<double>& getCoefs() const { return _coefs; }
        std::string repr() const;

    private:
        const std::vector<double> _coefs;
        const double _R_outer, _R_inner;
        mutable MatrixXd _coef_array;
        mutable MatrixXd _coef_array_gradx;
        mutable MatrixXd _coef_array_grady;
        mutable bool _coef_array_ready{false};
        mutable bool _coef_array_grad_ready{false};
    };

    namespace zernike {
        unsigned long long nCr(unsigned n, unsigned k);
        std::vector<double> binomial(double a, double b, unsigned n);

        std::pair<unsigned int,int> noll_to_zern(unsigned j);
        double _zern_norm(int n, int m);
        std::vector<long long> _zern_rho_coefs(unsigned n, int m);
        double h(int m, int j, double eps);
        std::vector<double> Q(int m, int j, double eps);
        double Q0(int m, int j, double eps);
        std::vector<double> _annular_zern_rho_coefs(int n, int m, double eps);
    }
}

#endif
