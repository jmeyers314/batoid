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
        RayVector(std::vector<Ray> _rays) : rays(_rays) {}

        // std::vector forwarding
        typename std::vector<Ray>::const_reference operator[](std::vector<Ray>::size_type i) const
            { return rays[i]; }
        typename std::vector<Ray>::iterator begin() noexcept { return rays.begin(); }
        typename std::vector<Ray>::iterator end() noexcept { return rays.end(); }
        typename std::vector<Ray>::size_type size() const noexcept { return rays.size(); }

        // data
        std::vector<Ray> rays;
    };

    std::vector<double> phaseMany(const std::vector<Ray>&, const Vector3d& r, double t);
    std::vector<std::complex<double>> amplitudeMany(const std::vector<Ray>&, const Vector3d& r, double t);
    std::complex<double> sumAmplitudeMany(const std::vector<Ray>&, const Vector3d& r, double t);
    std::vector<Ray> propagatedToTimesMany(const std::vector<Ray>&, const std::vector<double>& t);
    void propagateInPlaceMany(std::vector<Ray>&, const std::vector<double>& t);
}

#endif
