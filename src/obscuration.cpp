#include "obscuration.h"
#include "vec2.h"
#include "ray.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

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

    std::ostream& operator<<(std::ostream& os, const Obscuration& o) {
        return os << o.repr();
    }


    ObscCircle::ObscCircle(double radius, double x0, double y0) :
        _radius(radius), _x0(x0), _y0(y0) {}

    bool ObscCircle::contains(double x, double y) const {
        return std::hypot(x-_x0, y-_y0) < _radius;
    }

    bool ObscCircle::operator==(const Obscuration& rhs) const {
        if (const ObscCircle* other = dynamic_cast<const ObscCircle*> (&rhs)) {
            return _radius == other->_radius &&
                   _x0 == other->_x0 &&
                   _y0 == other->_y0;
        } else return false;
    }

    std::string ObscCircle::repr() const {
        std::ostringstream oss;
        oss << "ObscCircle("
            << _radius;
        if (_x0 != 0.0 || _y0 != 0.0) {
            oss << ", " << _x0
                << ", " << _y0;
        }
        oss << ")";
        return oss.str();
    }


    ObscAnnulus::ObscAnnulus(double inner, double outer, double x0, double y0) :
        _inner(inner), _outer(outer), _x0(x0), _y0(y0) {}

    bool ObscAnnulus::contains(double x, double y) const {
        double h = std::hypot(x-_x0, y-_y0);
        return (_inner <= h) && (h < _outer);
    }

    bool ObscAnnulus::operator==(const Obscuration& rhs) const {
        if (const ObscAnnulus* other = dynamic_cast<const ObscAnnulus*> (&rhs)) {
            return _inner == other->_inner &&
                   _outer == other->_outer &&
                   _x0 == other->_x0 &&
                   _y0 == other->_y0;
        } else return false;
    }

    std::string ObscAnnulus::repr() const {
        std::ostringstream oss;
        oss << "ObscAnnulus("
            << _inner << ", "
            << _outer;
        if (_x0 != 0.0 || _y0 != 0.0) {
            oss << ", " << _x0
                << ", " << _y0;
        }
        oss << ")";
        return oss.str();
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

    bool ObscRectangle::operator==(const Obscuration& rhs) const {
        if (const ObscRectangle* other = dynamic_cast<const ObscRectangle*> (&rhs)) {
            return _width == other->_width &&
                   _height == other->_height &&
                   _x0 == other->_x0 &&
                   _y0 == other->_y0 &&
                   _theta == other->_theta;
        } else return false;
    }

    std::string ObscRectangle::repr() const {
        std::ostringstream oss;
        oss << "ObscRectangle("
            << _width << ", "
            << _height;
        if (_x0 != 0.0 || _y0 != 0.0 || _theta != 0.0) {
            oss << ", " << _x0
                << ", " << _y0;
            if (_theta != 0.0)
                oss << ", " << _theta;
        }
        oss << ")";
        return oss.str();
    }



    ObscRay::ObscRay(double w, double th, double x0, double y0) :
        _width(w), _theta(th), _x0(x0), _y0(y0),
        _rect(ObscRectangle(x0 + 100*std::cos(th)/2,
                            y0 + 100*std::sin(th)/2,
                            100, w, th)) {}

    bool ObscRay::contains(double x, double y) const {
        return _rect.contains(x, y);
    }

    bool ObscRay::operator==(const Obscuration& rhs) const {
        if (const ObscRay* other = dynamic_cast<const ObscRay*> (&rhs)) {
            return _width == other->_width &&
                   _theta == other->_theta &&
                   _x0 == other->_x0 &&
                   _y0 == other->_y0;
        } else return false;
    }

    std::string ObscRay::repr() const {
        std::ostringstream oss;
        oss << "ObscRay("
            << _width << ", "
            << _theta;
        if (_x0 != 0.0 || _y0 != 0.0) {
            oss << ", " << _x0
                << ", " << _y0;
        }
        oss << ")";
        return oss.str();
    }


    ObscNegation::ObscNegation(const std::shared_ptr<Obscuration> original) :
        _original(original) {}

    bool ObscNegation::contains(double x, double y) const {
        return !_original->contains(x, y);
    }

    bool ObscNegation::operator==(const Obscuration& rhs) const {
        if (const ObscNegation* other = dynamic_cast<const ObscNegation*> (&rhs)) {
            return *_original == *other->_original;
        } else return false;
    }

    std::string ObscNegation::repr() const {
        std::ostringstream oss;
        oss << "ObscNegation("
            << *_original << ")";
        return oss.str();
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

    bool ObscUnion::operator==(const Obscuration& rhs) const {
        if (const ObscUnion* other = dynamic_cast<const ObscUnion*> (&rhs)) {
            return std::equal(
                _obscVec.begin(), _obscVec.end(), other->_obscVec.begin(),
                [](std::shared_ptr<Obscuration> a, std::shared_ptr<Obscuration> b){
                    return *a == *b;
                });
        } else return false;
    }

    std::string ObscUnion::repr() const {
        std::ostringstream oss;
        oss << "ObscUnion([";
        size_t i=0;
        for(; i< _obscVec.size()-1; i++)
            oss << *_obscVec[i] << ", ";
        oss << *_obscVec[i] << "])";
        return oss.str();
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

    bool ObscIntersection::operator==(const Obscuration& rhs) const {
        if (const ObscIntersection* other = dynamic_cast<const ObscIntersection*> (&rhs)) {
            return std::equal(
                _obscVec.begin(), _obscVec.end(), other->_obscVec.begin(),
                [](std::shared_ptr<Obscuration> a, std::shared_ptr<Obscuration> b){
                    return *a == *b;
                });
        } else return false;
    }

    std::string ObscIntersection::repr() const {
        std::ostringstream oss;
        oss << "ObscIntersection([";
        size_t i=0;
        for(; i< _obscVec.size()-1; i++)
            oss << *_obscVec[i] << ", ";
        oss << *_obscVec[i] << "])";
        return oss.str();
    }
}
