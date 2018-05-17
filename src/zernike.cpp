#include "zernike.h"
#include "LRUCache11.hpp"


namespace batoid {
    Zernike::Zernike(std::vector<double> coefs, double R_outer, double R_inner) :
        _coefs(coefs), _R_outer(R_outer), _R_inner(R_inner) {}

    double Zernike::sag(double x, double y) const {
        if (!_coef_array_ready) {
            _coef_array_ready = true;
        }
        return 0.0;
    }

    Vector3d Zernike::normal(double x, double y) const {
        return {0,0,1};
    }

    Ray Zernike::intersect(const Ray& r) const {
        return Ray{};
    }

    void Zernike::intersectInPlace(Ray& r) const {
        return;
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

        std::vector<double> binomial(double a, double b, unsigned n) {
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

        std::pair<unsigned int,int> noll_to_zern(unsigned int j) {
            static std::vector<unsigned int> _noll_n{0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4};
            static std::vector<int> _noll_m{0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4};
            unsigned int n;
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

        double _zern_norm(unsigned int n, int m) {
            if (m == 0) {
                return 1./std::sqrt(n+1.);
            } else {
                return 1./std::sqrt(2*n+2.);
            }
        }

        std::vector<long long> _zern_rho_coefs(unsigned int n, int m) {
            unsigned int kmax{(n-int(std::abs(m)))/2};
            std::vector<long long> A(n+1, 0.0);
            int pm = +1;
            for(unsigned int k=0; k<=kmax; k++, pm *= -1) {
                A[n-2*k] = pm * long(nCr(n-k, k) * nCr(n-2*k, kmax-k));
            }
            return A;
        }

        // Following 3 functions from
        // "Zernike annular polynomials for imaging systems with annular pupils"
        // Mahajan (1981) JOSA Vol. 71, No. 1.

        using iid_t = std::tuple<int, int, double>;
        lru11::Cache<iid_t,double> hCache(1024);

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

        lru11::Cache<iid_t,std::vector<double>> QCache(1024);

        std::vector<double> Q(int m, int j, double eps) {
            std::vector<double> result(j+1, 0.0);
            iid_t key = std::make_tuple(m, j, eps);
            if (QCache.tryGet(key, result)) {
                return result;
            }

            if (m == 0) {  // Eqn. (A4)
                std::vector<double> avec{_annular_zern_rho_coefs(2*j, 0, eps)};
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

        lru11::Cache<iid_t, double> Q0Cache(1024);

        double Q0(int m, int j, double eps) {
            double result = 0.0;
            iid_t key = std::make_tuple(m, j, eps);
            if (Q0Cache.tryGet(key, result)) {
                return result;
            }

            if (m == 0) {  // Eqn. (A4)
                result = _annular_zern_rho_coefs(2*j, 0, eps)[0];
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

        lru11::Cache<iid_t, std::vector<double>> azrcCache(1024);

        std::vector<double> _annular_zern_rho_coefs(int n, int m, double eps) {

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
                std::vector<long long> _zrc{_zern_rho_coefs(n, 0)};
                std::vector<double> coefs{_zrc.begin(), _zrc.end()};
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
    }
}
