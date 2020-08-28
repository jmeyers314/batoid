#include "obscuration.h"
#include <new>
#include <cmath>


namespace batoid {

    #pragma omp declare target

        Obscuration::Obscuration() :
            _devPtr(nullptr)
        {}

        Obscuration::~Obscuration() {}


        ObscCircle::ObscCircle(double radius, double x0, double y0) :
            Obscuration(), _radius(radius), _x0(x0), _y0(y0)
        {}

        ObscCircle::~ObscCircle() {}

        bool ObscCircle::contains(double x, double y) const {
            return std::hypot(x-_x0, y-_y0) < _radius;
        }

    #pragma omp end declare target

    Obscuration* ObscCircle::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscCircle(_radius, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }


    #pragma omp declare target

        ObscAnnulus::ObscAnnulus(double inner, double outer, double x0, double y0) :
            _inner(inner), _outer(outer), _x0(x0), _y0(y0)
        {}

        ObscAnnulus::~ObscAnnulus() {}

        bool ObscAnnulus::contains(double x, double y) const {
            double h = std::hypot(x-_x0, y-_y0);
            return (_inner <= h) && (h < _outer);
        }

    #pragma omp end declare target

    Obscuration* ObscAnnulus::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscAnnulus(_inner, _outer, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }

    // ObscRectangle::ObscRectangle(double w, double h, double x0, double y0, double th) :
    //     _width(w), _height(h), _x0(x0), _y0(y0), _theta(th)
    // {
    //     _A = {-_width/2, -_height/2};
    //     _B = {-_width/2, +_height/2};
    //     _C = {+_width/2, +_height/2};
    //     double sth = std::sin(_theta);
    //     double cth = std::cos(_theta);
    //     Matrix2d R;
    //     R << cth, -sth, sth, cth;
    //     Vector2d center(_x0, _y0);
    //     _A = R*_A + center;
    //     _B = R*_B + center;
    //     _C = R*_C + center;
    //     _AB = _B - _A;
    //     _ABAB = _AB.squaredNorm();
    //     _BC = _C - _B;
    //     _BCBC = _BC.squaredNorm();
    // }
    //
    // bool ObscRectangle::contains(double x, double y) const {
    //     Vector2d M(x, y);
    //     Vector2d AM(M - _A);
    //     Vector2d BM(M - _B);
    //     double ABAM = AM.dot(_AB);
    //     double BCBM = BM.dot(_BC);
    //     return (0 <= ABAM) && (ABAM <= _ABAB) && (0 <= BCBM) && (BCBM <= _BCBC);
    // }
    //
    // bool ObscRectangle::operator==(const Obscuration& rhs) const {
    //     if (const ObscRectangle* other = dynamic_cast<const ObscRectangle*>(&rhs)) {
    //         return _width == other->_width &&
    //                _height == other->_height &&
    //                _x0 == other->_x0 &&
    //                _y0 == other->_y0 &&
    //                _theta == other->_theta;
    //     } else return false;
    // }
    //
    // std::string ObscRectangle::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscRectangle("
    //         << _width << ", "
    //         << _height;
    //     if (_x0 != 0.0 || _y0 != 0.0 || _theta != 0.0) {
    //         oss << ", " << _x0
    //             << ", " << _y0;
    //         if (_theta != 0.0)
    //             oss << ", " << _theta;
    //     }
    //     oss << ")";
    //     return oss.str();
    // }
    //
    //
    // ObscRay::ObscRay(double w, double th, double x0, double y0) :
    //     _width(w), _theta(th), _x0(x0), _y0(y0),
    //     _rect(ObscRectangle(10, w,
    //                         x0 + 10*std::cos(th)/2,
    //                         y0 + 10*std::sin(th)/2,
    //                         th)) { }
    //
    // bool ObscRay::contains(double x, double y) const {
    //     return _rect.contains(x, y);
    // }
    //
    // bool ObscRay::operator==(const Obscuration& rhs) const {
    //     if (const ObscRay* other = dynamic_cast<const ObscRay*>(&rhs)) {
    //         return _width == other->_width &&
    //                _theta == other->_theta &&
    //                _x0 == other->_x0 &&
    //                _y0 == other->_y0;
    //     } else return false;
    // }
    //
    // std::string ObscRay::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscRay("
    //         << _width << ", "
    //         << _theta;
    //     if (_x0 != 0.0 || _y0 != 0.0) {
    //         oss << ", " << _x0
    //             << ", " << _y0;
    //     }
    //     oss << ")";
    //     return oss.str();
    // }
    //
    //
    // ObscPolygon::ObscPolygon(const std::vector<double>& xp, const std::vector<double>& yp) :
    //     _xp(xp), _yp(yp) {}
    //
    // bool ObscPolygon::contains(double x, double y) const {
    //     int size = _xp.size();
    //
    //     double x1 = _xp[0];
    //     double y1 = _yp[0];
    //     double xinters = 0.0;
    //     bool inside = false;
    //     for (int i=1; i<=size; i++) {
    //         double x2 = _xp[i % size];
    //         double y2 = _yp[i % size];
    //         if (y > std::min(y1,y2)) {
    //             if (y <= std::max(y1,y2)) {
    //                 if (x <= std::max(x1,x2)) {
    //                     if (y1 != y2) {
    //                         xinters = (y-y1)*(x2-x1)/(y2-y1)+x1;
    //                     }
    //                     if (x1 == x2 or x <= xinters) {
    //                         inside = !inside;
    //                     }
    //                 }
    //             }
    //         }
    //         x1 = x2;
    //         y1 = y2;
    //     }
    //     return inside;
    // }
    //
    // bool ObscPolygon::operator==(const Obscuration& rhs) const {
    //     if (const ObscPolygon* other = dynamic_cast<const ObscPolygon*>(&rhs)) {
    //         return _xp == other->_xp && _yp == other->_yp;
    //     } else return false;
    // }
    //
    // std::string ObscPolygon::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscPolygon([";
    //     std::string separator;
    //     for(auto& _xp0 : _xp) {
    //         oss << separator << _xp0;
    //         separator = ", ";
    //     }
    //     oss << "], [";
    //     separator = "";
    //     for(auto& _yp0 : _yp) {
    //         oss << separator << _yp0;
    //         separator = ", ";
    //     }
    //     oss << "])";
    //     return oss.str();
    // }
    //
    //
    // ObscNegation::ObscNegation(const std::shared_ptr<Obscuration> original) :
    //     _original(original) {}
    //
    // bool ObscNegation::contains(double x, double y) const {
    //     return !_original->contains(x, y);
    // }
    //
    // bool ObscNegation::operator==(const Obscuration& rhs) const {
    //     if (const ObscNegation* other = dynamic_cast<const ObscNegation*>(&rhs)) {
    //         return *_original == *other->_original;
    //     } else return false;
    // }
    //
    // std::string ObscNegation::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscNegation("
    //         << *_original << ")";
    //     return oss.str();
    // }
    //
    //
    // ObscUnion::ObscUnion(const std::vector<std::shared_ptr<Obscuration>> obscVec) :
    //     _obscVec(sortedObscurations(obscVec)) {}
    //
    // bool ObscUnion::contains(double x, double y) const {
    //     bool ret = false;
    //     for(const auto& obscPtr : _obscVec) {
    //         ret = ret || obscPtr->contains(x, y);
    //     }
    //     return ret;
    // }
    //
    // bool ObscUnion::operator==(const Obscuration& rhs) const {
    //     if (const ObscUnion* other = dynamic_cast<const ObscUnion*>(&rhs)) {
    //         if (_obscVec.size() != other->_obscVec.size()) return false;
    //         return std::equal(
    //             _obscVec.begin(), _obscVec.end(), other->_obscVec.begin(),
    //             [](std::shared_ptr<Obscuration> a, std::shared_ptr<Obscuration> b){
    //                 return *a == *b;
    //             });
    //     } else return false;
    // }
    //
    // std::string ObscUnion::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscUnion([";
    //     size_t i=0;
    //     for(; i<_obscVec.size()-1; i++)
    //         oss << *_obscVec[i] << ", ";
    //     oss << *_obscVec[i] << "])";
    //     return oss.str();
    // }
    //
    //
    // ObscIntersection::ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> obscVec) :
    //     _obscVec(sortedObscurations(obscVec)) {}
    //
    // bool ObscIntersection::contains(double x, double y) const {
    //     bool ret = true;
    //     for(const auto& obscPtr : _obscVec) {
    //         ret = ret && obscPtr->contains(x, y);
    //     }
    //     return ret;
    // }
    //
    // bool ObscIntersection::operator==(const Obscuration& rhs) const {
    //     if (const ObscIntersection* other = dynamic_cast<const ObscIntersection*>(&rhs)) {
    //         if (_obscVec.size() != other->_obscVec.size()) return false;
    //         return std::equal(
    //             _obscVec.begin(), _obscVec.end(), other->_obscVec.begin(),
    //             [](std::shared_ptr<Obscuration> a, std::shared_ptr<Obscuration> b){
    //                 return *a == *b;
    //             });
    //     } else return false;
    // }
    //
    // std::string ObscIntersection::repr() const {
    //     std::ostringstream oss;
    //     oss << "ObscIntersection([";
    //     size_t i=0;
    //     for(; i<_obscVec.size()-1; i++)
    //         oss << *_obscVec[i] << ", ";
    //     oss << *_obscVec[i] << "])";
    //     return oss.str();
    // }
}
