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
        Ray(double x, double y, double z, double vx, double vy, double vz,
            double t, double w, double f, bool vignetted);
        Ray(Vector3d _r, Vector3d _v, double _t, double w, double f, bool vignetted);
        Ray(const Ray& _ray) = default;
        Ray(bool failed);
        Ray() = default;

        Vector3d r;  // reference position
        Vector3d v;  // "velocity" Vector3d, really v/c
        double t; // reference time, really c*t0
        double wavelength; // in vacuum, in meters
        double flux;
        bool vignetted;
        bool failed;

        Vector3d positionAtTime(double _t) const;
        Ray propagatedToTime(double _t) const;
        void propagateInPlace(double _t);
        bool operator==(const Ray&) const;
        bool operator!=(const Ray&) const;

        void setFail() { failed=true; }
        void clearFail() { failed=false; }

        std::string repr() const;

        Vector3d k() const { return 2 * PI * v / wavelength / v.squaredNorm(); }
        double omega() const { return 2 * PI / wavelength; }  // really omega/c.
        double phase(const Vector3d& _r, double _t) const;
        std::complex<double> amplitude(const Vector3d& _r, double _t) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Ray& _r) {
        return os << _r.repr();
    }

}

#endif
