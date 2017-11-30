#include "obscuration.h"
#include "vec2.h"
#include "ray.h"
#include "utils.h"
#include <cmath>

namespace batoid {
    Ray Obscuration::obscure(const Ray& ray) const {
        if (ray.failed || ray.isVignetted) return ray;
        if (contains(ray.p0.x, ray.p0.y))
            return Ray(ray.p0, ray.v, ray.t0, ray.wavelength, true);
        else
            return ray;
    }

    void Obscuration::obscureInPlace(Ray& ray) const {
        if (ray.failed || ray.isVignetted) return;
        if (contains(ray.p0.x, ray.p0.y))
            ray.isVignetted = true;
    }

    std::vector<Ray> Obscuration::obscure(const std::vector<Ray>& rays) const {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray)
            {
                if (ray.failed) return ray;
                if (contains(ray.p0.x, ray.p0.y))
                    return Ray(ray.p0, ray.v, ray.t0, ray.wavelength, true);
                else
                    return Ray(ray.p0, ray.v, ray.t0, ray.wavelength, ray.isVignetted);
            },
            2000
        );
        return result;
    }

    void Obscuration::obscureInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(
            rays.begin(), rays.end(),
            [this](Ray& r){ obscureInPlace(r); },
            2000
        );
    }

    ObscCircle::ObscCircle(double radius, double x0, double y0) :
        _radius(radius), _x0(x0), _y0(y0) {}

    bool ObscCircle::contains(double x, double y) const {
        return std::hypot(x-_x0, y-_y0) < _radius;
    }

    ObscAnnulus::ObscAnnulus(double inner, double outer, double x0, double y0) :
        _inner(inner), _outer(outer), _x0(x0), _y0(y0) {}

    bool ObscAnnulus::contains(double x, double y) const {
        double h = std::hypot(x-_x0, y-_y0);
        return (_inner <= h) && (h < _outer);
    }

    ObscRectangle::ObscRectangle(double w, double h, double x0, double y0, double th) :
        _width(w), _height(h), _x0(x0), _y0(y0), _theta(th)
    {
        _A = {-_width/2, -_height/2};
        _B = {-_width/2, +_height/2};
        _C = {+_width/2, +_height/2};
        double sth = std::sin(_theta);
        double cth = std::cos(_theta);
        Rot2 R{{{cth, -sth, sth, cth}}};
        Vec2 center(_x0, _y0);
        _A = RotVec(R, _A) + center;
        _B = RotVec(R, _B) + center;
        _C = RotVec(R, _C) + center;
        _AB = _B - _A;
        _ABAB = DotProduct(_AB, _AB);
        _BC = _C - _B;
        _BCBC = DotProduct(_BC, _BC);
    }

    bool ObscRectangle::contains(double x, double y) const {
        Vec2 M(x, y);
        Vec2 AM(M - _A);
        Vec2 BM(M - _B);
        double ABAM(DotProduct(AM, _AB));
        double BCBM(DotProduct(BM, _BC));
        return (0 <= ABAM) && (ABAM <= _ABAB) && (0 <= BCBM) && (BCBM <= _BCBC);
    }

    ObscRay::ObscRay(double w, double th, double x0, double y0) :
        _width(w), _theta(th), _x0(x0), _y0(y0),
        _rect(ObscRectangle(x0 + 100*std::cos(th)/2,
                            y0 + 100*std::sin(th)/2,
                            100, w, th)) {}

    bool ObscRay::contains(double x, double y) const {
        return _rect.contains(x, y);
    }

    ObscUnion::ObscUnion(const std::vector<std::shared_ptr<Obscuration>> obscVec) :
        _obscVec(obscVec) {}

    bool ObscUnion::contains(double x, double y) const {
        bool ret = false;
        for(const auto& obscPtr : _obscVec) {
            ret = ret || obscPtr->contains(x, y);
        }
        return ret;
    }

    ObscIntersection::ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> obscVec) :
        _obscVec(obscVec) {}

    bool ObscIntersection::contains(double x, double y) const {
        bool ret = true;
        for(const auto& obscPtr : _obscVec) {
            ret = ret && obscPtr->contains(x, y);
        }
        return ret;
    }

    ObscNegation::ObscNegation(const std::shared_ptr<Obscuration> original) :
        _original(original) {}

    bool ObscNegation::contains(double x, double y) const {
        return !_original->contains(x, y);
    }
}
