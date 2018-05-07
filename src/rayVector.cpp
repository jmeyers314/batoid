#include "rayVector.h"
#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    std::vector<double> RayVector::phase(const Vector3d& r, double t) {
        auto result = std::vector<double>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.phase(r, t); }
        );
        return result;
    }

    std::vector<std::complex<double>> RayVector::amplitude(const Vector3d& r, double t) {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return result;
    }

    std::complex<double> RayVector::sumAmplitude(const Vector3d& r, double t) {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return std::accumulate(result.begin(), result.end(), std::complex<double>(0,0));
    }

    RayVector RayVector::propagatedToTime(double t) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.propagatedToTime(t); }
        );
        return result;
    }

    void RayVector::propagateInPlace(double t) {
        parallel_for_each(rays.begin(), rays.end(),
            [=](Ray& ray)
                { ray.propagateInPlace(t); }
        );
    }

    RayVector RayVector::trimVignetted() {
        RayVector result;
        result.rays.reserve(rays.size());
        std::copy_if(
            rays.begin(),
            rays.end(),
            std::back_inserter(result.rays),
            [](const Ray& r){return !r.isVignetted;}
        );
        return result;
    }

    void RayVector::trimVignettedInPlace() {
        rays.erase(
            std::remove_if(
                rays.begin(),
                rays.end(),
                [](const Ray& r){ return r.failed || r.isVignetted; }
            ),
            rays.end()
        );
    }

}
