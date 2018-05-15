#include "sphere.h"
#include "utils.h"

namespace batoid {
    Sphere::Sphere(double R) : _R(R) {}

    double Sphere::sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-std::sqrt(1-(x*x + y*y)/_R/_R));
        return 0.0;
    }

    Vector3d Sphere::normal(double x, double y) const {
        if (_R == 0.0)
            return Vector3d(0,0,1);
        double r = std::sqrt(x*x + y*y);
        if (r == 0)
            return Vector3d(0,0,1);
        double dzdr1 = dzdr(r);
        return Vector3d(-dzdr1*x/r, -dzdr1*y/r, 1).normalized();
    }

    bool Sphere::timeToIntersect(const Ray& r, double& t) const {
        double vr2 = r.v[0]*r.v[0] + r.v[1]*r.v[1];
        double vz2 = r.v[2]*r.v[2];
        double vrr0 = r.v[0]*r.p0[0] + r.v[1]*r.p0[1];
        double r02 = r.p0[0]*r.p0[0] + r.p0[1]*r.p0[1];
        double z0term = (r.p0[2]-_R);

        // Quadratic equation coefficients
        double a = vz2 + vr2;
        double b = 2*r.v[2]*z0term + 2*vrr0;
        double c = z0term*z0term - _R*_R + r02;

        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        // Should probably check the solutions here since we obtained the quadratic
        // formula above by squaring both sides of an equation.
        if (n == 0) {
            return false;
        } else if (n == 1) {
            if (r1 < 0)
                return false;
            t = r1;
        } else {
            if (r1 < 0) {
                if (r2 < 0)
                    return false;
                else
                    t = r2;
            } else
                t = std::min(r1, r2);
        }
        t += r.t0;
        return true;
    }

    Ray Sphere::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    std::string Sphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Sphere(" << _R << ")";
        return oss.str();
    }

    void Sphere::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntersect(r, t)) {
            r.failed=true;
            return;
        }
        r.p0 = r.positionAtTime(t);
        r.t0 = t;
        return;
    }

    double Sphere::dzdr(double r) const {
        double rat = r/_R;
        return rat/std::sqrt(1-rat*rat);
    }

}
