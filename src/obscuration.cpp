#include "obscuration.h"
#include "ray.h"
#include "rayVector.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

namespace batoid {
    std::vector<std::shared_ptr<Obscuration>> Obscuration::sortedObscurations(const std::vector<std::shared_ptr<Obscuration>> obscVec) {
        std::vector<std::shared_ptr<Obscuration>> result(obscVec);
        std::sort(
            result.begin(), result.end(),
            [](std::shared_ptr<Obscuration> a, std::shared_ptr<Obscuration> b)
            { return a->repr() < b->repr(); }
        );
        return result;
    }

    Ray Obscuration::obscure(const Ray& ray) const {
        if (ray.failed || ray.vignetted) return ray;
        if (contains(ray.r[0], ray.r[1]))
            return Ray(ray.r, ray.v, ray.t, ray.wavelength, ray.flux, true);
        else
            return ray;
    }

    void Obscuration::obscureInPlace(Ray& ray) const {
        if (ray.failed || ray.vignetted) return;
        if (contains(ray.r[0], ray.r[1]))
            ray.vignetted = true;
    }

    RayVector Obscuration::obscure(const RayVector& rv) const {
        std::vector<Ray> result(rv.size());
        parallelTransform(rv.cbegin(), rv.cend(), result.begin(),
            [this](const Ray& ray)
            {
                if (ray.failed) return ray;
                if (contains(ray.r[0], ray.r[1]))
                    return Ray(ray.r, ray.v, ray.t, ray.wavelength, ray.flux, true);
                else
                    return Ray(ray.r, ray.v, ray.t, ray.wavelength, ray.flux, ray.vignetted);
            }
        );
        return RayVector(std::move(result), rv.getWavelength());
    }

    void Obscuration::obscureInPlace(RayVector& rv) const {
        parallel_for_each(
            rv.begin(), rv.end(),
            [this](Ray& r){ obscureInPlace(r); }
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
        if (const ObscCircle* other = dynamic_cast<const ObscCircle*>(&rhs)) {
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

  ObscEllipse::ObscEllipse(double dx, double dy, double x0, double y0) :
    _dx(dx), _dy(dy), _x0(x0), _y0(y0) {}

  bool ObscEllipse::contains(double x, double y) const {
    return std::hypot((x-_x0)/_dx, (y-_y0)/_dy) < 1.;
  }

  bool ObscEllipse::operator==(const Obscuration & rhs) const {
    if(const ObscEllipse * other = dynamic_cast<const ObscEllipse*>(&rhs)) {
      return _dx == other->_dx &&
	_dy == other->_dy &&
	_x0 == other->_x0 &&
	_y0 == other->_y0;
    }
    else
      return false;
  }
  
  std::string ObscEllipse::repr() const {
    std::ostringstream oss;
    oss << "ObscEllipse("
	<< _dx << " " << _dy;
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
        if (const ObscAnnulus* other = dynamic_cast<const ObscAnnulus*>(&rhs)) {
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
        Matrix2d R;
        R << cth, -sth, sth, cth;
        Vector2d center(_x0, _y0);
        _A = R*_A + center;
        _B = R*_B + center;
        _C = R*_C + center;
        _AB = _B - _A;
        _ABAB = _AB.squaredNorm();
        _BC = _C - _B;
        _BCBC = _BC.squaredNorm();
    }

    bool ObscRectangle::contains(double x, double y) const {
        Vector2d M(x, y);
        Vector2d AM(M - _A);
        Vector2d BM(M - _B);
        double ABAM = AM.dot(_AB);
        double BCBM = BM.dot(_BC);
        return (0 <= ABAM) && (ABAM <= _ABAB) && (0 <= BCBM) && (BCBM <= _BCBC);
    }

    bool ObscRectangle::operator==(const Obscuration& rhs) const {
        if (const ObscRectangle* other = dynamic_cast<const ObscRectangle*>(&rhs)) {
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
        _rect(ObscRectangle(10, w,
                            x0 + 10*std::cos(th)/2,
                            y0 + 10*std::sin(th)/2,
                            th)) { }

    bool ObscRay::contains(double x, double y) const {
        return _rect.contains(x, y);
    }

    bool ObscRay::operator==(const Obscuration& rhs) const {
        if (const ObscRay* other = dynamic_cast<const ObscRay*>(&rhs)) {
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
        if (const ObscNegation* other = dynamic_cast<const ObscNegation*>(&rhs)) {
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
        _obscVec(sortedObscurations(obscVec)) {}

    bool ObscUnion::contains(double x, double y) const {
        bool ret = false;
        for(const auto& obscPtr : _obscVec) {
            ret = ret || obscPtr->contains(x, y);
        }
        return ret;
    }

    bool ObscUnion::operator==(const Obscuration& rhs) const {
        if (const ObscUnion* other = dynamic_cast<const ObscUnion*>(&rhs)) {
            if (_obscVec.size() != other->_obscVec.size()) return false;
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
        for(; i<_obscVec.size()-1; i++)
            oss << *_obscVec[i] << ", ";
        oss << *_obscVec[i] << "])";
        return oss.str();
    }


    ObscIntersection::ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> obscVec) :
        _obscVec(sortedObscurations(obscVec)) {}

    bool ObscIntersection::contains(double x, double y) const {
        bool ret = true;
        for(const auto& obscPtr : _obscVec) {
            ret = ret && obscPtr->contains(x, y);
        }
        return ret;
    }

    bool ObscIntersection::operator==(const Obscuration& rhs) const {
        if (const ObscIntersection* other = dynamic_cast<const ObscIntersection*>(&rhs)) {
            if (_obscVec.size() != other->_obscVec.size()) return false;
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
        for(; i<_obscVec.size()-1; i++)
            oss << *_obscVec[i] << ", ";
        oss << *_obscVec[i] << "])";
        return oss.str();
    }
}
