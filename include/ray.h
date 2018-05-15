#ifndef batoid_ray_h
#define batoid_ray_h

#include <sstream>
#include <string>
#include <complex>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    const double PI = 3.14159265358979323846;
    struct Ray {
        Ray(double x0, double y0, double z0, double vx, double vy, double vz,
            double t, double w, bool isVignetted);
        Ray(Vector3d _p0, Vector3d _v, double t, double w, bool isVignetted);
        Ray(const Ray& _ray) = default;
        Ray(bool failed);
        Ray() = default;

        Vector3d p0; // reference position
        Vector3d v;  // "velocity" Vector3d, really v/c
        double t0; // reference time, really c*t0
        double wavelength; // in vacuum, in meters
        bool isVignetted;
        bool failed;

        Vector3d positionAtTime(double t) const;
        Ray propagatedToTime(double t) const;
        void propagateInPlace(double t);
        bool operator==(const Ray&) const;
        bool operator!=(const Ray&) const;

        void setFail() { failed=true; }
        void clearFail() { failed=false; }

        std::string repr() const;

        Vector3d k() const { return 2 * PI * v / wavelength / v.squaredNorm(); }
        double omega() const { return 2 * PI / wavelength; }
        double phase(const Vector3d& r, double t) const;
        std::complex<double> amplitude(const Vector3d& r, double t) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Ray& r) {
        return os << r.repr();
    }

}

#endif
