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
        // Ray(std::array<double,3> _p0, std::array<double,3> _v,
        //     double t, double w, bool isVignetted);
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
