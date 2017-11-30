#include "paraboloid.h"
#include "utils.h"
#include "except.h"
#include <cmath>

namespace batoid {

    Paraboloid::Paraboloid(double R) : _R(R) {}

    std::string Paraboloid::repr() const {
        std::ostringstream oss(" ");
        oss << "Paraboloid(" << _R << ")";
        return oss.str();
    }

    double Paraboloid::sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(2*_R);
        return 0.0;
    }

    Vec3 Paraboloid::normal(double x, double y) const {
        if (_R != 0.0)
            return Vec3(-x/_R, -y/_R, 1).UnitVec3();
        return Vec3(0,0,1);
    }

    bool Paraboloid::timeToIntercept(const Ray& r, double& t) const {
        double a = (r.v.x*r.v.x + r.v.y*r.v.y)/2/_R;
        double b = (r.p0.x*r.v.x + r.p0.y*r.v.y)/_R - r.v.z;
        double c = (r.p0.x*r.p0.x + r.p0.y*r.p0.y)/2/_R - r.p0.z;
        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        if (n == 0)
            return false;
        else if (n == 1) {
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

    Ray Paraboloid::intercept(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntercept(r, t))
            return Ray(true);
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    Intersection Paraboloid::intersect(const Ray& r) const {
        if (r.failed)
            return Intersection(true);
        double t;
        if (!timeToIntercept(r, t))
            return Intersection(true);
        Vec3 point = r.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
    }

    void Paraboloid::interceptInPlace(Ray& r) const {
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

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid& p) {
        return os << p.repr();
    }

}
