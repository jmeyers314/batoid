#include "rayVector.h"
#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    RayVector::RayVector(
        const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
        const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
        const std::vector<double>& t, const std::vector<double>& w, const std::vector<bool>& vignetted
    ) {
        rays.reserve(x.size());
        bool wSame{true};
        double w0{w[0]};
        for(int i=0; i<x.size(); i++) {
            rays.push_back(Ray(x[i], y[i], z[i], vx[i], vy[i], vz[i], t[i], w[i], vignetted[i]));
            if (w[i] != w0) wSame = false;
        }
        if (wSame) wavelength=w0;
    }

    std::string RayVector::repr() const {
        std::ostringstream oss("RayVector([", std::ios_base::ate);
        oss << rays[0];
        for(int i=1; i<rays.size(); i++) {
            oss << ", " << rays[i];
        }
        oss << ']';
        if (!std::isnan(wavelength))
            oss << ", wavelength=" << wavelength;
        oss << ')';
        return oss.str();
    }

    std::vector<double> RayVector::phase(const Vector3d& r, double t) const {
        auto result = std::vector<double>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.phase(r, t); }
        );
        return result;
    }

    std::vector<std::complex<double>> RayVector::amplitude(const Vector3d& r, double t) const {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return result;
    }

    std::complex<double> RayVector::sumAmplitude(const Vector3d& r, double t) const {
        auto result = std::vector<std::complex<double>>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return std::accumulate(result.begin(), result.end(), std::complex<double>(0,0));
    }

    std::vector<Vector3d> RayVector::positionAtTime(double t) const {
        auto result = std::vector<Vector3d>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.positionAtTime(t); }
        );
        return result;
    }

    RayVector RayVector::propagatedToTime(double t) const {
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

    RayVector RayVector::trimVignetted() const {
        RayVector result;
        result.rays.reserve(rays.size());
        std::copy_if(
            rays.begin(),
            rays.end(),
            std::back_inserter(result.rays),
            [](const Ray& r){return !r.vignetted;}
        );
        return result;
    }

    void RayVector::trimVignettedInPlace() {
        rays.erase(
            std::remove_if(
                rays.begin(),
                rays.end(),
                [](const Ray& r){ return r.failed || r.vignetted; }
            ),
            rays.end()
        );
    }

    RayVector concatenateRayVectors(const std::vector<RayVector>& rvs) {
        int n = std::accumulate(
            rvs.begin(), rvs.end(), 0,
            [](int s, const RayVector& rv){ return s + rv.size(); }
        );
        std::vector<Ray> out;
        out.reserve(n);

        double wavelength = rvs[0].wavelength;
        for (const auto& rv: rvs) {
            if (wavelength != rv.wavelength)
                wavelength = NAN;
            out.insert(out.end(), rv.rays.cbegin(), rv.rays.cend());
        }
        return RayVector(out, wavelength);
    }
}
