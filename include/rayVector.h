#ifndef batoid_rayVector_h
#define batoid_rayVector_h

#include "ray.h"
#include <complex>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    struct RayVector {
        RayVector() {}
        RayVector(const RayVector& _rv) : rays(_rv.rays) {}
        RayVector(RayVector&& _rv) : rays(std::move(_rv.rays)) {}
        RayVector(const std::vector<Ray>& _rays) : rays(_rays) {}
        RayVector(std::vector<Ray>&& _rays) : rays(std::move(_rays)) {}

        // std::vector forwarding
        typename std::vector<Ray>::const_reference operator[](std::vector<Ray>::size_type i) const
            { return rays[i]; }
        typename std::vector<Ray>::iterator begin() noexcept { return rays.begin(); }
        typename std::vector<Ray>::iterator end() noexcept { return rays.end(); }
        typename std::vector<Ray>::size_type size() const noexcept { return rays.size(); }

        // methods
        std::vector<double> phase(const Vector3d& r, double t);
        std::vector<std::complex<double>> amplitude(const Vector3d& r, double t);
        std::complex<double> sumAmplitude(const Vector3d& r, double t);
        RayVector propagatedToTime(double t);
        void propagateInPlace(double t);
        RayVector trimVignetted();
        void trimVignettedInPlace();

        // data
        std::vector<Ray> rays;
    };
}

#endif
