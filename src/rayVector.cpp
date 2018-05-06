#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    std::vector<double> phaseMany(const std::vector<Ray>& rays, const Vector3d& r, double t) {
        auto result = std::vector<double>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.phase(r, t); }
        );
        return result;
    }

    std::vector<std::complex<double>> amplitudeMany(const std::vector<Ray>& rays, const Vector3d& r, double t) {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return result;
    }

    std::complex<double> sumAmplitudeMany(const std::vector<Ray>& rays, const Vector3d& r, double t) {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return std::accumulate(result.begin(), result.end(), std::complex<double>(0,0));
    }

    std::vector<Ray> propagatedToTimesMany(const std::vector<Ray>& rays, const std::vector<double>& ts) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), ts.cbegin(), result.begin(),
            [](const Ray& ray, double t)
                { return ray.propagatedToTime(t); }
        );
        return result;
    }

    void propagateInPlaceMany(std::vector<Ray>& rays, const std::vector<double>& ts) {
        parallel_for_each(rays.begin(), rays.end(), ts.begin(),
            [](Ray& ray, double t)
                { ray.propagateInPlace(t); }
        );
    }
}
