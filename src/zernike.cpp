#include "zernike.h"
#include "LRUCache11.hpp"
#include "solve.h"


namespace batoid {
    Zernike::Zernike(std::vector<double> coefs, double R_outer, double R_inner) :
        _coefs(coefs), _R_outer(R_outer), _R_inner(R_inner) {}

    double Zernike::sag(double x, double y) const {
        if (!_coef_array_ready) {
            std::lock_guard<std::mutex> lock(_mtx);
            if (!_coef_array_ready) {
                std::vector<MatrixXd> coef_array_vec(
                    zernike::noll_coef_array_xy(_coefs.size()-1, _R_inner/_R_outer)
                );
                _coef_array = MatrixXd::Zero(coef_array_vec[0].rows(), coef_array_vec[0].cols());
                for(int j=1; j<_coefs.size(); j++) {
                    _coef_array += _coefs[j]*coef_array_vec[j-1];
                }
                // Need to adjust for R_outer != 1.0 here
                for(int i=0; i<_coef_array.rows(); i++) {
                    for(int j=0; j<_coef_array.cols(); j++) {
                        _coef_array(i, j) /= std::pow(_R_outer, i+j);
                    }
                }
                _coef_array_ready = true;
            }
        }
        return zernike::horner2d(x, y, _coef_array);
    }

    Vector3d Zernike::normal(double x, double y) const {
        return {0,0,1};
    }

    Ray Zernike::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Zernike::intersectInPlace(Ray& r) const {
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

    class ZernikeResidual {
    public:
        ZernikeResidual(const Zernike& z, const Ray& r) : _z(z), _r(r) {}
        double operator()(double t) const {
            Vector3d p = _r.positionAtTime(t);
            return _z.sag(p(0), p(1)) - p(2);
        }
    private:
        const Zernike& _z;
        const Ray& _r;
    };

    bool Zernike::timeToIntersect(const Ray& r, double& t) const {
        // Guess that 0.0 is a good initial estimate
        // (Could maybe use Z4 to make a paraboloid?)
        t = 0.0;
        ZernikeResidual resid(*this, r);
        Solve<ZernikeResidual> solve(resid, t, t+1e-2);
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

    std::string Zernike::repr() const {
        return "Zernike";
    }

    namespace zernike {

        //https://www.quora.com/What-are-some-efficient-algorithms-to-compute-nCr/answer/Vladimir-Novakovski
        unsigned long long nCr(unsigned int n, unsigned int k) {
            std::vector<std::pair<unsigned, unsigned>> prime_factors;
            std::vector<bool> composite(n+2, false);
            composite[0] = composite[1] = true;
            unsigned q, m;
            int total_prime_power;
            for(unsigned p=2; p<=n; p++) {
                if (composite[p]) continue;
                q = p;
                m = 1;
                total_prime_power = 0;
                std::vector<unsigned> prime_power(n+1, 0);
                while(true) {
                    prime_power[q] = prime_power[m] + 1;
                    unsigned r = q;
                    if (q <= k)
                        total_prime_power -= prime_power[q];
                    if (q > (n-k))
                        total_prime_power += prime_power[q];
                    m += 1;
                    q += p;
                    if (q > n) break;
                    composite[q] = true;
                }
                prime_factors.emplace_back(p, total_prime_power);
            }
            unsigned long long result{1};
            for(const auto& pf : prime_factors) {
                for(int i=0; i<pf.second; i++)
                    result *= pf.first;
            }
            return result;
        }

        std::vector<double> binomial(double a, double b, int n) {
            std::vector<double> result;
            result.reserve(n+1);
            double b_over_a = b/a;
            double c = std::pow(a, n);
            for(int i=0; i<n+1; i++) {
                result.push_back(c);
                c *= b_over_a;
                c *= n-i;
                c /= i+1;
            }
            return result;
        }

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

        std::pair<int,int> noll_to_zern(int j) {
            static std::vector<int> _noll_n{0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4};
            static std::vector<int> _noll_m{0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4};
            int n;
            int m, pm;
            while (_noll_n.size() <= j) {
                n = _noll_n.back() + 1;
                _noll_n.insert(_noll_n.end(), n+1, n);
                if (n%2==0) {
                    _noll_m.push_back(0);
                    m = 2;
                } else {
                    m = 1;
                }
                pm = ((n/2)%2==0) ? +1 : -1;
                while (m <= n) {
                    _noll_m.push_back(pm*m);
                    _noll_m.push_back(-pm*m);
                    m += 2;
                }
            }
            return {_noll_n[j], _noll_m[j]};
        }

        double zern_norm(int n, int m) {
            if (m == 0) {
                return 1./std::sqrt(n+1.);
            } else {
                return 1./std::sqrt(2*n+2.);
            }
        }

        std::vector<double> zern_rho_coefs(int n, int m) {
            int kmax{(n-int(std::abs(m)))/2};
            std::vector<double> A(n+1, 0.0);
            int pm = +1;
            for(int k=0; k<=kmax; k++, pm *= -1) {
                A[n-2*k] = pm * (double)(nCr(n-k, k) * nCr(n-2*k, kmax-k));
            }
            return A;
        }

        // Following 3 functions from
        // "Zernike annular polynomials for imaging systems with annular pupils"
        // Mahajan (1981) JOSA Vol. 71, No. 1.

        using iid_t = std::tuple<int, int, double>;
        using id_t = std::tuple<int, double>;
        lru11::Cache<iid_t,double,std::mutex> hCache(1024);

        double h(int m, int j, double eps) {
            double result;
            iid_t key = std::make_tuple(m, j, eps);
            if (hCache.tryGet(key, result)) {
                return result;
            }

            if (m == 0) return (1-eps*eps)/(2*(2*j+1));  // Eqn. (A5)
            // Eqn. (A14)
            double num = -(2*(2*j+2*m-1)) * Q0(m-1, j+1, eps);
            double den = (j+m)*(1-eps*eps) * Q0(m-1, j, eps);
            result = num/den * h(m-1, j, eps);

            hCache.insert(key, result);
            return result;
        }

        lru11::Cache<iid_t,std::vector<double>,std::mutex> QCache(1024);

        std::vector<double> Q(int m, int j, double eps) {
            std::vector<double> result(j+1, 0.0);
            iid_t key = std::make_tuple(m, j, eps);
            if (QCache.tryGet(key, result)) {
                return result;
            }

            if (m == 0) {  // Eqn. (A4)
                std::vector<double> avec{annular_zern_rho_coefs(2*j, 0, eps)};
                for(int i=0; i<avec.size(); i+=2)
                    result[i/2] = avec[i];
            } else {  // Eqn. (A13)
                double num = 2*(2*j+2*m-1) * h(m-1, j, eps);
                double den = (j+m)*(1-eps*eps)*Q0(m-1, j, eps);
                for (int i=0; i<j+1; i++) {
                    std::vector<double> qq{Q(m-1, i, eps)};
                    double qq0 = qq[0];
                    for(int k=0; k<qq.size(); k++) {
                        qq[k] *= qq0;
                    }
                    for(int k=0; k<i+1; k++) {
                        result[k] += qq[k]/h(m-1, i, eps);
                    }
                }
                for(int k=0; k<result.size(); k++) {
                    result[k] *= num/den;
                }
            }

            QCache.insert(key, result);
            return result;
        }

        lru11::Cache<iid_t,double,std::mutex> Q0Cache(1024);
        double Q0(int m, int j, double eps) {
            double result = 0.0;
            iid_t key = std::make_tuple(m, j, eps);
            if (Q0Cache.tryGet(key, result)) {
                return result;
            }

            if (m == 0) {  // Eqn. (A4)
                result = annular_zern_rho_coefs(2*j, 0, eps)[0];
            } else {  // Eqn. (A13)
                double num = 2*(2*j+2*m-1) * h(m-1, j, eps);
                double den = (j+m)*(1-eps*eps)*Q0(m-1, j, eps);
                for (int i=0; i<j+1; i++) {
                    double qq = Q0(m-1, i, eps);
                    result += qq*qq/h(m-1, i, eps);
                }
                result *= num/den;
            }

            Q0Cache.insert(key, result);
            return result;
        }

        lru11::Cache<iid_t,std::vector<double>,std::mutex> azrcCache(1024);
        std::vector<double> annular_zern_rho_coefs(int n, int m, double eps) {
            std::vector<double> result(n+1, 0.0);
            iid_t key = std::make_tuple(n, m, eps);
            if (azrcCache.tryGet(key, result)) {
                return result;
            }

            m = std::abs(m);
            if (m == 0) {  // Eqn. (18)
                double norm = 1./(1-eps*eps);
                // R[n, m=0, eps](r^2) = R[n, m=0, eps=0]((r^2 - eps^2)/(1 - eps^2))
                // Implement this by retrieving R[n, 0] coefficients of (r^2)^k and
                // multiplying in the binomial (in r^2) expansion of ((r^2 - eps^2)/(1 - eps^2))^k
                std::vector<double> coefs{zern_rho_coefs(n, 0)};
                for(int i=0; i<n+1; i++) {
                    if (i%2==1) continue;
                    int j = i/2;
                    std::vector<double> bin{binomial(-eps*eps, 1.0, j)};
                    for(int k=0; 2*k<(i+1); k++) {
                        result[2*k] += coefs[i]*std::pow(norm, j)*bin[k];
                    }
                }
            } else if (m == n) {  // Eqn (25)
                double sum = 0;
                for(int i=0; i<n+1; i++) {
                    sum += std::pow(eps*eps, i);
                }
                double norm = 1./std::sqrt(sum);
                result[n] = norm;
            } else {  // Eqn (A1)
                int j = (n-m)/2;
                double norm = std::sqrt((1-eps*eps)/(2*(2*j+m+1) * h(m,j,eps)));
                std::vector<double> Qvec{Q(m, j, eps)};
                for(int k=0; 2*k+m<n+1; k++) {
                    result[m+2*k] = norm*Qvec[k];
                }
            }

            azrcCache.insert(key, result);
            return result;
        }

        MatrixXcd zern_coef_array(int n, int m, double eps, std::pair<int,int> shape) {
            int am = std::abs(m);
            MatrixXcd result = MatrixXcd::Zero(shape.first, shape.second);
            std::vector<double> coefs;
            if (0.0 < eps && eps < 1.0) {
                coefs = annular_zern_rho_coefs(n, m, eps);
            } else if (eps == 0.0) {
                coefs = zern_rho_coefs(n, m);
            } else {
                throw std::runtime_error("Invalid obscuration");
            }
            double norm = zern_norm(n, m);
            for(int i=0; i<coefs.size(); i++) {
                coefs[i] /= norm;
            }
            for(int i=0; am+2*i<coefs.size(); i++) {
                result(i, am) = (m >= 0) ? coefs[am+2*i] : std::complex<double>(0,-1) * coefs[am+2*i];
            }
            return result;
        }

        lru11::Cache<id_t,std::vector<MatrixXcd>,std::mutex> ncaCache(1024);
        std::vector<MatrixXcd> noll_coef_array(int jmax, double eps) {
            std::vector<MatrixXcd> result;
            id_t key = std::make_tuple(jmax, eps);
            if (ncaCache.tryGet(key, result)) {
                return result;
            }

            int maxn = noll_to_zern(jmax).first;
            std::pair<int,int> shape(maxn/2+1, maxn+1);
            result.reserve(jmax);
            for(int j=1; j<jmax+1; j++) {
                std::pair<int,int> nm = noll_to_zern(j);
                result.emplace_back(zern_coef_array(nm.first, nm.second, eps, shape));
            }

            ncaCache.insert(key, result);
            return result;
        }

        MatrixXcd xy_contribution(int rho2_power, int rho_power, std::pair<int,int> shape) {
            MatrixXcd result = MatrixXcd::Zero(shape.first, shape.second);
            result(0,0) = 1;
            for(;rho2_power > 0; rho2_power--) {
                MatrixXcd tmp = MatrixXcd::Zero(shape.first, shape.second);
                for(int i=0; i<shape.first-2; i++) {
                    for(int j=0; j<shape.second-2; j++) {
                        std::complex<double> val = result(i, j);
                        tmp(i+2, j) += val;
                        tmp(i, j+2) += val;
                    }
                }
                result = tmp;
            }
            for(;rho_power > 0; rho_power--) {
                MatrixXcd tmp = MatrixXcd::Zero(shape.first, shape.second);
                for(int i=0; i<shape.first-1; i++) {
                    for(int j=0; j<shape.second-1; j++) {
                        std::complex<double> val = result(i, j);
                        tmp(i+1, j) += val;
                        tmp(i, j+1) += val*std::complex<double>(0,1);
                    }
                }
                result = tmp;
            }
            return result;
        }

        MatrixXd rrsq_to_xy(const MatrixXcd& coefs, std::pair<int,int> shape) {
            MatrixXd result = MatrixXd::Zero(shape.first, shape.second);
            for(int i=0; i<coefs.rows(); i++) {
                for(int j=0; j<coefs.cols(); j++) {
                    if (coefs(i, j) == 0.0) continue;
                    result += (coefs(i, j) * xy_contribution(i, j, shape)).real();
                }
            }
            return result;
        }

        MatrixXd xycoef_gradx(const MatrixXd& coefs, std::pair<int,int> shape) {
            MatrixXd result = MatrixXd::Zero(shape.first, shape.second);
            for(int i=0; i<shape.first; i++) {
                for(int j=0; j<shape.second; j++) {
                    if (i>0) result(i-1, j) = coefs(i, j)*i;
                }
            }
            return result;
        }

        MatrixXd xycoef_grady(const MatrixXd& coefs, std::pair<int,int> shape) {
            MatrixXd result = MatrixXd::Zero(shape.first, shape.second);
            for(int i=0; i<shape.first; i++) {
                for(int j=0; j<shape.second; j++) {
                    if (j>0) result(i, j-1) = coefs(i, j)*j;
                }
            }
            return result;
        }

        lru11::Cache<id_t,std::vector<MatrixXd>,std::mutex> ncaxyCache(1024);
        std::vector<MatrixXd> noll_coef_array_xy(int jmax, double eps) {
            std::vector<MatrixXd> result;
            id_t key = std::make_tuple(jmax, eps);
            if (ncaxyCache.tryGet(key, result)) {
                return result;
            }

            int maxn = noll_to_zern(jmax).first;
            std::pair<int,int> shape(maxn+1, maxn+1);

            std::vector<MatrixXcd> nca{noll_coef_array(jmax, eps)};
            for(int j=1; j<jmax+1; j++) {
                result.push_back(rrsq_to_xy(nca[j-1], shape));
            }

            ncaxyCache.insert(key, result);
            return result;
        }
    } // namespace zernike
} // namespace batoid
