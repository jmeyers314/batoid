#include "sphere.h"
#include "utils.h"
#include "except.h"

namespace batoid {
    Sphere::Sphere(double R) : _R(R) {}

    double Sphere::sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-std::sqrt(1-(x*x + y*y)/_R/_R));
        return 0.0;
    }

    Vec3 Sphere::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0) return Vec3(0,0,1);
        double dzdr1 = dzdr(r);
        return Vec3(-dzdr1*x/r, -dzdr1*y/r, 1).UnitVec3();
    }

    bool Sphere::timeToIntercept(const Ray& r, double& t) const {
        double vr2 = r.v.x*r.v.x + r.v.y*r.v.y;
        double vz2 = r.v.z*r.v.z;
        double vrr0 = r.v.x*r.p0.x + r.v.y*r.p0.y;
        double r02 = r.p0.x*r.p0.x + r.p0.y*r.p0.y;
        double z0term = (r.p0.z-_R);

        // Quadratic equation coefficients
        double a = vz2 + vr2;
        double b = 2*r.v.z*z0term + 2*vrr0;
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
            } else {
                if (r2 < 0)
                    t = r1;
                else
                    t = std::min(r1, r2);
            }
        }
        t += r.t0;
        return true;
    }

    Ray Sphere::intercept(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntercept(r, t))
            return Ray(true);
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    Intersection Sphere::intersect(const Ray &r) const {
        if (r.failed)
            return Intersection(true);
        double t;
        if (!timeToIntercept(r, t))
            return Intersection(true);
        Vec3 point = r.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
    }

    std::string Sphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Sphere(" << _R << ")";
        return oss.str();
    }

    void Sphere::interceptInPlace(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntercept(r, t)) {
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

    inline std::ostream& operator<<(std::ostream& os, const Sphere& q) {
        return os << q.repr();
    }

}
