#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    Ray::Ray(double x0, double y0, double z0, double vx, double vy, double vz, double t=0.0,
             double w=0.0, bool isV=false) :
        p0(Vector3d(x0, y0, z0)), v(Vector3d(vx, vy, vz)), t0(t),
        wavelength(w), isVignetted(isV), failed(false) {}

    Ray::Ray(Vector3d _p0, Vector3d _v, double t=0.0, double w=0.0, bool isV=false) :
        p0(_p0), v(_v), t0(t), wavelength(w), isVignetted(isV), failed(false) {}

    // Ray::Ray(std::array<double,3> _p0, std::array<double,3> _v,
    //          double t=0.0, double w=0.0, bool isV=false) :
    //     p0(Vector3d(_p0[0], _p0[1], _p0[2])), v(Vector3d(_v[0], _v[1], _v[2])),
    //     t0(t), wavelength(w), isVignetted(isV), failed(false) {}

    Ray::Ray(const bool failed) :
        p0(Vector3d::Zero()), v(Vector3d::Zero()), t0(0.0), wavelength(0.0), isVignetted(true), failed(true) {}

    std::string Ray::repr() const {
        std::ostringstream oss("Ray(", std::ios_base::ate);
        if(failed)
            oss << "failed)";
        else {
            oss << p0 << ", " << v;
            if (t0 != 0.0) oss << ", t=" << t0;
            if (wavelength != 0.0) oss << ", wavelength=" << wavelength;
            if (isVignetted) oss << ", isVignetted=True";
            oss << ")";
        }
        return oss.str();
    }

    Vector3d Ray::positionAtTime(const double t) const {
        return p0+v*(t-t0);
    }

    Ray Ray::propagatedToTime(const double t) const {
        return Ray(positionAtTime(t), v, t, wavelength, isVignetted);
    }

    void Ray::propagateInPlace(const double t) {
        p0 += v*(t-t0);
        t0 = t;
    }

    bool Ray::operator==(const Ray& other) const {
        // All failed rays are equal
        if (failed)
            return other.failed;
        if (other.failed)
            return false;
        return (p0 == other.p0) &&
               (v == other.v) &&
               (t0 == other.t0) &&
               (wavelength == other.wavelength) &&
               (isVignetted == other.isVignetted);
    }

    bool Ray::operator!=(const Ray& other) const {
        return !(*this == other);
    }

    double Ray::phase(const Vector3d& r, double t) const {
        return k().dot(r-p0) - (t-t0)*omega();
    }

    std::complex<double> Ray::amplitude(const Vector3d& r, double t) const {
        return std::exp(std::complex<double>(0, 1)*phase(r, t));
    }

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
