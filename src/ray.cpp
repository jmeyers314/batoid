#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    Ray::Ray(double x, double y, double z, double vx, double vy, double vz, double _t=0.0,
             double _wavelength=0.0, bool _vignetted=false) :
        r(Vector3d(x, y, z)), v(Vector3d(vx, vy, vz)), t(_t),
        wavelength(_wavelength), vignetted(_vignetted), failed(false) {}

    Ray::Ray(Vector3d _r, Vector3d _v, double _t=0.0, double _wavelength=0.0,
             bool _vignetted=false) :
        r(_r), v(_v), t(_t), wavelength(_wavelength), vignetted(_vignetted), failed(false) {}

    Ray::Ray(const bool failed) :
        r(Vector3d::Zero()), v(Vector3d::Zero()), t(0.0), wavelength(0.0), vignetted(true),
        failed(true) {}

    std::string Ray::repr() const {
        std::ostringstream oss("Ray(", std::ios_base::ate);
        if(failed)
            oss << "failed=True)";
        else {
            oss << "[" << r[0] << "," << r[1] << "," << r[2] << "],["
                << v[0] << "," << v[1] << "," << v[2] << "]";
            if (t != 0.0) oss << ", t=" << t;
            if (wavelength != 0.0) oss << ", wavelength=" << wavelength;
            if (vignetted) oss << ", vignetted=True";
            oss << ")";
        }
        return oss.str();
    }

    Vector3d Ray::positionAtTime(const double _t) const {
        return r+v*(_t-t);
    }

    Ray Ray::propagatedToTime(const double _t) const {
        return Ray(positionAtTime(_t), v, _t, wavelength, vignetted);
    }

    void Ray::propagateInPlace(const double _t) {
        r += v*(_t-t);
        t = _t;
    }

    bool Ray::operator==(const Ray& other) const {
        // All failed rays are equal
        if (failed)
            return other.failed;
        if (other.failed)
            return false;
        return (r == other.r) &&
               (v == other.v) &&
               (t == other.t) &&
               (wavelength == other.wavelength) &&
               (vignetted == other.vignetted);
    }

    bool Ray::operator!=(const Ray& other) const {
        return !(*this == other);
    }

    double Ray::phase(const Vector3d& _r, double _t) const {
        return k().dot(_r-r) - (_t-t)*omega();
    }

    std::complex<double> Ray::amplitude(const Vector3d& _r, double _t) const {
        return std::exp(std::complex<double>(0, 1)*phase(_r, _t));
    }
}
