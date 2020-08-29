#include "obscuration.h"
#include <new>
#include <cmath>
#include <algorithm>


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


    #pragma omp declare target

        ObscRectangle::ObscRectangle(double w, double h, double x0, double y0, double th) :
            Obscuration(), _width(w), _height(h), _x0(x0), _y0(y0), _theta(th),
            _sth(std::sin(th)), _cth(std::cos(th))
        {}

        ObscRectangle::~ObscRectangle() {}

        bool ObscRectangle::contains(double x, double y) const {
            double xp = (x-_x0)*_cth + (y-_y0)*_sth;
            double yp = -(x-_x0)*_sth + (y-_y0)*_cth;
            return (xp > -_width/2 && xp < _width/2 && yp > -_height/2 && yp < _height/2);
        }

    #pragma omp end declare target

    Obscuration* ObscRectangle::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscRectangle* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscRectangle(_width, _height, _x0, _y0, _theta);
        }
        _devPtr = ptr;
        return ptr;
    }


    #pragma omp declare target

        ObscRay::ObscRay(double w, double th, double x0, double y0) :
            Obscuration(), _width(w), _theta(th), _x0(x0), _y0(y0),
            _sth(std::sin(th)), _cth(std::cos(th))
        {}

        ObscRay::~ObscRay() {}

        bool ObscRay::contains(double x, double y) const {
            double xp = (x-_x0)*_cth + (y-_y0)*_sth;
            double yp = -(x-_x0)*_sth + (y-_y0)*_cth;
            return (xp > 0.0 && yp > -_width/2 && yp < _width/2);
        }

    #pragma omp end declare target

    Obscuration* ObscRay::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscRay* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscRay(_width, _theta, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }


    #pragma omp declare target

        ObscPolygon::ObscPolygon(const double* xp, const double* yp, const size_t size) :
            Obscuration(), _xp(_copyArr(xp, size)), _yp(_copyArr(yp, size)), _size(size)
        {}

        ObscPolygon::~ObscPolygon() {
            delete[] _xp;
            delete[] _yp;
        }

        bool ObscPolygon::contains(double x, double y) const {
            double x1 = _xp[0];
            double y1 = _yp[0];
            double xinters = 0.0;
            bool inside = false;
            for (int i=1; i<=_size; i++) {
                double x2 = _xp[i % _size];
                double y2 = _yp[i % _size];
                if (y > std::min(y1,y2)) {
                    if (y <= std::max(y1,y2)) {
                        if (x <= std::max(x1,x2)) {
                            if (y1 != y2) {
                                xinters = (y-y1)*(x2-x1)/(y2-y1)+x1;
                            }
                            if (x1 == x2 or x <= xinters) {
                                inside = !inside;
                            }
                        }
                    }
                }
                x1 = x2;
                y1 = y2;
            }
            return inside;
        }

        double* ObscPolygon::_copyArr(const double* arr, const size_t size) {
            double* out = new double[size];
            for(int i=0; i<size; i++)
                out[i] = arr[i];
            return out;
        }

    #pragma omp end declare target

    Obscuration* ObscPolygon::getDevPtr() const {
        if (!_devPtr) {
            Obscuration* ptr;
            #pragma omp target map(from:ptr) map(to:_xp[:_size], _yp[:_size])
            {
                ptr = new ObscPolygon(_xp, _yp, _size);
            }
            _devPtr = ptr;
        }
        return _devPtr;
    }


    #pragma omp declare target

        ObscNegation::ObscNegation(Obscuration* original) :
            Obscuration(), _original(original)
        {}

        ObscNegation::~ObscNegation() {}

        bool ObscNegation::contains(double x, double y) const {
            return !_original->contains(x, y);
        }

    #pragma omp end declare target

    Obscuration* ObscNegation::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscNegation* ptr;
        Obscuration* originalDevPtr = _original->getDevPtr();

        #pragma omp target map(from:ptr) is_device_ptr(originalDevPtr)
        {
            ptr = new ObscNegation(originalDevPtr);
        }
        _devPtr = ptr;
        return ptr;
    }


    #pragma omp declare target

        ObscUnion::ObscUnion(Obscuration** obscs, size_t nobsc) :
            Obscuration(), _nobsc(nobsc)
        {
            _obscs = new Obscuration*[_nobsc];
            for (int i=0; i<_nobsc; i++) {
                _obscs[i] = obscs[i];
            }
        }

        ObscUnion::~ObscUnion() {}

        bool ObscUnion::contains(double x, double y) const {
            bool ret = false;
            for (int i=0; i<_nobsc; i++) {
                ret |= _obscs[i]->contains(x, y);
            }
            return ret;
        }

    #pragma omp end declare target

    Obscuration* ObscUnion::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration** devPtrs = new Obscuration*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            devPtrs[i] = _obscs[i]->getDevPtr();
        }
        ObscUnion* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr) map(to:devPtrs[:_nobsc])
        {
            ptr = new ObscUnion(devPtrs, _nobsc);
        }
        _devPtr = ptr;
        return ptr;
    }


    #pragma omp declare target

        ObscIntersection::ObscIntersection(Obscuration** obscs, size_t nobsc) :
            Obscuration(), _nobsc(nobsc)
        {
            _obscs = new Obscuration*[_nobsc];
            for (int i=0; i<_nobsc; i++) {
                _obscs[i] = obscs[i];
            }
        }

        ObscIntersection::~ObscIntersection() {}

        bool ObscIntersection::contains(double x, double y) const {
            bool ret = true;
            for (int i=0; i<_nobsc; i++) {
                ret &= _obscs[i]->contains(x, y);
            }
            return ret;
        }

    #pragma omp end declare target

    Obscuration* ObscIntersection::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration** devPtrs = new Obscuration*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            devPtrs[i] = _obscs[i]->getDevPtr();
        }
        ObscIntersection* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr) map(to:devPtrs[:_nobsc])
        {
            ptr = new ObscIntersection(devPtrs, _nobsc);
        }
        _devPtr = ptr;
        return ptr;
    }
}
