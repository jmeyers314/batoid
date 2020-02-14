#ifndef batoid_rayVector_h
#define batoid_rayVector_h

#include "ray.h"
#include "coordsys.h"
#include <memory>
#include <cmath>
#include <complex>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    struct RayVector {
        RayVector() = default;
        RayVector(const RayVector& _rv) = default;
        RayVector(RayVector&& _rv) = default;
        RayVector& operator=(const RayVector& rv) = default;
        RayVector& operator=(RayVector&& rv) = default;

        RayVector(const std::vector<Ray>& rays, const CoordSys& coordSys) : _rays(rays), _coordSys(coordSys) {}
        RayVector(std::vector<Ray>&& rays, const CoordSys& coordSys) : _rays(std::move(rays)), _coordSys(coordSys) {}

        RayVector(const std::vector<Ray>& rays, const CoordSys& coordSys, double wavelength)
            : _rays(rays), _coordSys(coordSys), _wavelength(wavelength) {}
        RayVector(std::vector<Ray>&& rays, const CoordSys& coordSys, double wavelength)
            : _rays(std::move(rays)), _coordSys(coordSys), _wavelength(wavelength) {}

        RayVector(
            const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
            const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
            const std::vector<double>& t, const std::vector<double>& w,
            const std::vector<double>& flux, const std::vector<bool>& vignetted,
            const CoordSys& coordSys
        );

        // std::vector forwarding
        typename std::vector<Ray>::const_reference operator[](std::vector<Ray>::size_type i) const
            { return _rays[i]; }
        typename std::vector<Ray>::reference operator[](std::vector<Ray>::size_type i)
            { return _rays[i]; }
        typename std::vector<Ray>::iterator begin() noexcept { return _rays.begin(); }
        typename std::vector<Ray>::iterator end() noexcept { return _rays.end(); }
        typename std::vector<Ray>::const_iterator cbegin() const noexcept { return _rays.cbegin(); }
        typename std::vector<Ray>::const_iterator cend() const noexcept { return _rays.cend(); }
        typename std::vector<Ray>::size_type size() const noexcept { return _rays.size(); }
        typename std::vector<Ray>::reference front() { return _rays.front(); }
        typename std::vector<Ray>::const_reference front() const { return _rays.front(); }
        void push_back(const Ray& r) { _rays.push_back(r); }
        void push_back(Ray&& r) { _rays.push_back(r); }

        // note: no test on _wavelength.
        bool operator==(const RayVector& rhs) const {
            return _rays == rhs._rays;
        }
        bool operator!=(const RayVector& rhs) const {
            return _rays != rhs._rays;
        }

        // methods
        std::vector<Vector3d> positionAtTime(double t) const;
        RayVector propagatedToTime(double t) const;
        void propagateInPlace(double t);

        std::vector<double> phase(const Vector3d& r, double t) const;
        std::vector<std::complex<double>> amplitude(const Vector3d& r, double t) const;
        std::complex<double> sumAmplitude(const Vector3d& r, double t) const;
        RayVector trimVignetted(double minFlux) const;
        void trimVignettedInPlace(double minFlux);
        std::string repr() const;

        double getWavelength() const { return _wavelength; }
        const CoordSys& getCoordSys() const { return _coordSys; }
        const std::vector<Ray>& getRays() const { return _rays; }  // exposing just for pickle/hash

    private:
        std::vector<Ray> _rays;
        double _wavelength{NAN};  // If not NAN, then all wavelengths are this wavelength
        CoordSys _coordSys;
    };

    inline std::ostream& operator<<(std::ostream& os, const RayVector& rv) {
        return os << rv.repr();
    }

    RayVector concatenateRayVectors(const std::vector<RayVector>& rvs);
}

#endif
