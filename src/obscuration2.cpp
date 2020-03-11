#include "obscuration2.h"

#pragma omp declare target
namespace batoid {

    void Obscuration2::obscureInPlace(RayVector2& rv2) const {
        rv2.r.syncToDevice();
        rv2.vignetted.syncToDevice();
        rv2.failed.syncToDevice();
        size_t size = rv2.size;

        double* x = rv2.r.deviceData;
        double* y = rv2.r.deviceData + size;
        bool* failed = rv2.failed.deviceData;
        bool* vignetted = rv2.vignetted.deviceData;

        Obscuration2* devPtr = getDevPtr();
        #pragma omp target teams distribute parallel for is_device_ptr(x, y, failed, vignetted, devPtr)
        for(int i=0; i<size; i++) {
            if (!(failed[i] || vignetted[i])) {
                if (devPtr->contains(x[i], y[i]))
                    vignetted[i] = true;
            }
        }
    }



    ObscCircle2::ObscCircle2(double radius, double x0, double y0) :
        Obscuration2(), _radius(radius), _x0(x0), _y0(y0) {}

    Obscuration2* ObscCircle2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscCircle2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscCircle2(_radius, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscCircle2::contains(double x, double y) const {
        return (x-_x0)*(x-_x0) + (y-_y0)*(y-_y0) < _radius*_radius;
    }



    ObscAnnulus2::ObscAnnulus2(double inner, double outer, double x0, double y0) :
        Obscuration2(), _inner(inner), _outer(outer), _x0(x0), _y0(y0)
    {}

    Obscuration2* ObscAnnulus2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscAnnulus2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscAnnulus2(_inner, _outer, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscAnnulus2::contains(double x, double y) const {
        double r2 = (x-_x0)*(x-_x0) + (y-_y0)*(y-_y0);
        return (_inner*_inner <= r2) && (r2 <= _outer*_outer);
    }



    ObscRectangle2::ObscRectangle2(double w, double h, double x0, double y0, double th) :
        Obscuration2(), _width(w), _height(h), _x0(x0), _y0(y0), _theta(th)
    {
        double sth = std::sin(_theta);
        double cth = std::cos(_theta);

        _A0 = cth*(-_width/2) - sth*(-_height/2) + _x0;
        _A1 = sth*(-_width/2) + cth*(-_height/2) + _y0;

        _B0 = cth*(-_width/2) - sth*(+_height/2) + _x0;
        _B1 = sth*(-_width/2) + cth*(+_height/2) + _y0;

        double C0 = cth*(+_width/2) - sth*(+_height/2) + _x0;
        double C1 = sth*(+_width/2) + cth*(+_height/2) + _y0;

        _AB0 = _B0 - _A0;
        _AB1 = _B1 - _A1;
        _ABAB = _AB0*_AB0 + _AB1*_AB1;
        _BC0 = C0 - _B0;
        _BC1 = C1 - _B1;
        _BCBC = _BC0*_BC0 + _BC1*_BC1;
    }

    Obscuration2* ObscRectangle2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscRectangle2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscRectangle2(_width, _height, _x0, _y0, _theta);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscRectangle2::contains(double x, double y) const {
        double AM0 = x - _A0;
        double AM1 = y - _A1;
        double BM0 = x - _B0;
        double BM1 = y - _B1;
        double ABAM = AM0*_AB0 + AM1*_AB1;
        double BCBM = BM0*_BC0 + BM1*_BC1;
        return (0 <= ABAM) && (ABAM <= _ABAB) && (0 <= BCBM) && (BCBM <= _BCBC);
    }



    ObscRay2::ObscRay2(double w, double th, double x0, double y0) :
        Obscuration2(),
        _width(w), _theta(th), _x0(x0), _y0(y0),
        _rect(ObscRectangle2(10, w,
                             x0 + 10*std::cos(th)/2,
                             y0 + 10*std::sin(th)/2,
                             th)) { }

    Obscuration2* ObscRay2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscRay2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr)
        {
            ptr = new ObscRay2(_width, _theta, _x0, _y0);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscRay2::contains(double x, double y) const {
        return _rect.contains(x, y);
    }



    ObscNegation2::ObscNegation2(Obscuration2* original) :
        Obscuration2(), _original(original)
    {}

    Obscuration2* ObscNegation2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ObscNegation2* ptr;
        Obscuration2* originalDevPtr = _original->getDevPtr();

        // create device shadow instance
        #pragma omp target map(from:ptr) is_device_ptr(originalDevPtr)
        {
            ptr = new ObscNegation2(originalDevPtr);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscNegation2::contains(double x, double y) const {
        return !_original->contains(x, y);
    }



    ObscUnion2::ObscUnion2(Obscuration2** obscs, size_t nobsc) : Obscuration2(), _nobsc(nobsc) {
        _obscs = new Obscuration2*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            _obscs[i] = obscs[i];
        }
    }

    Obscuration2* ObscUnion2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration2** devPtrs = new Obscuration2*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            devPtrs[i] = _obscs[i]->getDevPtr();
        }
        ObscUnion2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr) map(to:devPtrs[:_nobsc])
        {
            ptr = new ObscUnion2(devPtrs, _nobsc);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscUnion2::contains(double x, double y) const {
        bool ret = false;
        for (int i=0; i<_nobsc; i++) {
            ret |= _obscs[i]->contains(x, y);
        }
        return ret;
    }



    ObscIntersection2::ObscIntersection2(Obscuration2** obscs, size_t nobsc) :
        Obscuration2(), _nobsc(nobsc)
    {
        _obscs = new Obscuration2*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            _obscs[i] = obscs[i];
        }
    }

    Obscuration2* ObscIntersection2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Obscuration2** devPtrs = new Obscuration2*[_nobsc];
        for (int i=0; i<_nobsc; i++) {
            devPtrs[i] = _obscs[i]->getDevPtr();
        }
        ObscIntersection2* ptr;
        // create device shadow instance
        #pragma omp target map(from:ptr) map(to:devPtrs[:_nobsc])
        {
            ptr = new ObscIntersection2(devPtrs, _nobsc);
        }
        _devPtr = ptr;
        return ptr;
    }

    bool ObscIntersection2::contains(double x, double y) const {
        bool ret = true;
        for (int i=0; i<_nobsc; i++) {
            ret &= _obscs[i]->contains(x, y);
        }
        return ret;
    }
}
#pragma omp end declare target
