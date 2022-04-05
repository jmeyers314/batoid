#include "obscuration.h"
#include <new>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    /////////////////
    // Obscuration //
    /////////////////

    Obscuration::Obscuration() {}

    Obscuration::~Obscuration() {}

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ////////////////
    // ObscCircle //
    ////////////////


    ObscCircle::ObscCircle(double radius, double x0, double y0) :
        _radius(radius), _x0(x0), _y0(y0)
    {}

    ObscCircle::~ObscCircle() {}

    bool ObscCircle::contains(double x, double y) const {
        return std::hypot(x-_x0, y-_y0) < _radius;
    }


    /////////////////
    // ObscAnnulus //
    /////////////////

    ObscAnnulus::ObscAnnulus(double inner, double outer, double x0, double y0) :
        _inner(inner), _outer(outer), _x0(x0), _y0(y0)
    {}

    ObscAnnulus::~ObscAnnulus() {}

    bool ObscAnnulus::contains(double x, double y) const {
        double h = std::hypot(x-_x0, y-_y0);
        return (_inner <= h) && (h < _outer);
    }


    ///////////////////
    // ObscRectangle //
    ///////////////////

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


    /////////////
    // ObscRay //
    /////////////

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


    /////////////////
    // ObscPolygon //
    /////////////////

    ObscPolygon::ObscPolygon(const double* xp, const double* yp, const size_t size) :
        Obscuration(), _xp(xp), _yp(yp), _size(size)
    {}

    ObscPolygon::~ObscPolygon() {}

    bool ObscPolygon::contains(double x, double y) const {
        double x1 = _xp[0];  // first point of segment
        double y1 = _yp[0];
        double xinters = 0.0;
        bool inside = false;
        for (int i=1; i<=_size; i++) {  // Loop over polygon edges
            double x2 = _xp[i % _size];  // second point of segment
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


    //////////////////
    // ObscNegation //
    //////////////////

    ObscNegation::ObscNegation(const Obscuration* original) :
        Obscuration(), _original(original)
    {}

    ObscNegation::~ObscNegation() {}

    bool ObscNegation::contains(double x, double y) const {
        return !_original->contains(x, y);
    }


    ///////////////
    // ObscUnion //
    ///////////////

    ObscUnion::ObscUnion(const Obscuration** obscs, size_t nobsc) :
        Obscuration(), _obscs(obscs), _nobsc(nobsc)
    {}

    ObscUnion::~ObscUnion() {}

    bool ObscUnion::contains(double x, double y) const {
        bool ret = false;
        for (int i=0; i<_nobsc; i++) {
            ret |= _obscs[i]->contains(x, y);
        }
        return ret;
    }


    //////////////////////
    // ObscIntersection //
    //////////////////////

    ObscIntersection::ObscIntersection(const Obscuration** obscs, size_t nobsc) :
        Obscuration(), _obscs(obscs), _nobsc(nobsc)
    {}

    ObscIntersection::~ObscIntersection() {}

    bool ObscIntersection::contains(double x, double y) const {
        bool ret = true;
        for (int i=0; i<_nobsc; i++) {
            ret &= _obscs[i]->contains(x, y);
        }
        return ret;
    }


    ///////////////////////
    // ObscurationHandle //
    ///////////////////////

    ObscurationHandle::ObscurationHandle() :
        _hostPtr(nullptr),
        _devicePtr(nullptr)
    {}

    ObscurationHandle::~ObscurationHandle() {}

    const Obscuration* ObscurationHandle::getPtr() const {
        #if defined(BATOID_GPU)
            return _devicePtr;
        #else
            return _hostPtr;
        #endif
    }

    const Obscuration* ObscurationHandle::getHostPtr() const {
        return _hostPtr;
    }


    //////////////////////
    // ObscCircleHandle //
    //////////////////////

    ObscCircleHandle::ObscCircleHandle(double radius, double x0, double y0) :
        ObscurationHandle()
    {
        _hostPtr = new ObscCircle(radius, x0, y0);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscCircle), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscCircle(radius, x0, y0);
            }
        #endif
    }

    ObscCircleHandle::~ObscCircleHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscCircle *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscCircle();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ///////////////////////
    // ObscAnnulusHandle //
    ///////////////////////

    ObscAnnulusHandle::ObscAnnulusHandle(double inner, double outer, double x0, double y0) :
        ObscurationHandle()
    {
        _hostPtr = new ObscAnnulus(inner, outer, x0, y0);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscAnnulus), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscAnnulus(inner, outer, x0, y0);
            }
        #endif
    }

    ObscAnnulusHandle::~ObscAnnulusHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscAnnulus *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscAnnulus();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    /////////////////////////
    // ObscRectangleHandle //
    /////////////////////////

    ObscRectangleHandle::ObscRectangleHandle(double w, double h, double x0, double y0, double th) :
        ObscurationHandle()
    {
        _hostPtr = new ObscRectangle(w, h, x0, y0, th);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscRectangle), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscRectangle(w, h, x0, y0, th);
            }
        #endif
    }

    ObscRectangleHandle::~ObscRectangleHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscRectangle *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscRectangle();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ///////////////////
    // ObscRayHandle //
    ///////////////////

    ObscRayHandle::ObscRayHandle(double w, double th, double x0, double y0) :
        ObscurationHandle()
    {
        _hostPtr = new ObscRay(w, th, x0, y0);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscRay), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscRay(w, th, x0, y0);
            }
        #endif
    }

    ObscRayHandle::~ObscRayHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscRay *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscRay();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ///////////////////////
    // ObscPolygonHandle //
    ///////////////////////

    ObscPolygonHandle::ObscPolygonHandle(const double* xp, const double* yp, const size_t size) :
        ObscurationHandle(),
        _xp(xp),
        _yp(yp),
        _size(size)
    {
        _hostPtr = new ObscPolygon(xp, yp, size);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscPolygon), omp_get_default_device());
            const size_t lcl_size = _size;
            const double* lcl_xp = _xp;
            const double* lcl_yp = _yp;
            #pragma omp target enter data map(to:lcl_xp[:lcl_size], lcl_yp[:lcl_size])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscPolygon(lcl_xp, lcl_yp, lcl_size);
            }
        #endif
    }

    ObscPolygonHandle::~ObscPolygonHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscPolygon *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscPolygon();
            // }
            #pragma omp target exit data \
                map(release:_xp[:_size], _yp[:_size])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ////////////////////////
    // ObscNegationHandle //
    ////////////////////////

    ObscNegationHandle::ObscNegationHandle(const ObscurationHandle* handle) :
        ObscurationHandle(),
        _original(handle->getPtr())
    {
        _hostPtr = new ObscNegation(handle->getHostPtr());
        #if defined(BATOID_GPU)
            const Obscuration* m_original = _original;
            auto alloc = omp_target_alloc(sizeof(ObscNegation), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc, m_original)
            {
                _devicePtr = new (alloc) ObscNegation(m_original);
            }
        #endif
    }

    ObscNegationHandle::~ObscNegationHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscNegation *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscNegation();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    /////////////////////
    // ObscUnionHandle //
    /////////////////////

    const Obscuration** ObscUnionHandle::_getObscs(
        const ObscurationHandle** handles, const size_t nobsc, bool host
    ) {
        auto out = new const Obscuration*[nobsc];
        for (size_t i=0; i<nobsc; i++) {
            out[i] = host ? handles[i]->getHostPtr() : handles[i]->getPtr();
        }
        return out;
    }

    ObscUnionHandle::ObscUnionHandle(const ObscurationHandle** handles, size_t nobsc) :
        ObscurationHandle(),
        _hostObscs(_getObscs(handles, nobsc, true)),
        _devObscs(_getObscs(handles, nobsc, false)),
        _nobsc(nobsc)
    {
        _hostPtr = new ObscUnion(_hostObscs, _nobsc);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscUnion), omp_get_default_device());
            const Obscuration** devO = _devObscs;
            size_t no = _nobsc;
            #pragma omp target enter data map(to:devO[:no])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscUnion(devO, no);
            }
        #endif
    }

    ObscUnionHandle::~ObscUnionHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscUnion *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscUnion();
            // }

            #pragma omp target exit data map(release:_devObscs[:_nobsc])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete[] _hostObscs;
        delete[] _devObscs;
        delete _hostPtr;
    }


    ////////////////////////////
    // ObscIntersectionHandle //
    ////////////////////////////

    const Obscuration** ObscIntersectionHandle::_getObscs(
        const ObscurationHandle** handles, const size_t nobsc, bool host
    ) {
        auto out = new const Obscuration*[nobsc];
        for (size_t i=0; i<nobsc; i++) {
            out[i] = host ? handles[i]->getHostPtr() : handles[i]->getPtr();
        }
        return out;
    }

    ObscIntersectionHandle::ObscIntersectionHandle(const ObscurationHandle** handles, size_t nobsc) :
        ObscurationHandle(),
        _hostObscs(_getObscs(handles, nobsc, true)),
        _devObscs(_getObscs(handles, nobsc, false)),
        _nobsc(nobsc)
    {
        _hostPtr = new ObscIntersection(_hostObscs, _nobsc);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ObscIntersection), omp_get_default_device());
            const Obscuration** devO = _devObscs;
            size_t no = _nobsc;
            #pragma omp target enter data map(to:devO[:no])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ObscIntersection(devO, no);
            }
        #endif
    }

    ObscIntersectionHandle::~ObscIntersectionHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<ObscIntersection *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~ObscIntersection();
            // }

            #pragma omp target exit data map(release:_devObscs[:_nobsc])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete[] _hostObscs;
        delete[] _devObscs;
        delete _hostPtr;
    }


    // // CPU only for now.
    // void ObscPolygon::containsGrid(
    //     const double* xgrid, const double* ygrid, bool* out, const size_t nx, const size_t ny
    // ) const {
    //     // xgrid is [nx], ygrid is [ny]
    //     // out is [ny, nx]

    //     // Compute polygon y min/max
    //     double ymin = _yp[0];
    //     double ymax = _yp[0];
    //     for (int k=1; k<_size; k++) {
    //         if (_yp[k] < ymin)
    //             ymin = _yp[k];
    //         if (_yp[k] > ymax)
    //             ymax = _yp[k];
    //     }

    //     // Proceed row by row
    //     std::vector<double> xinters;
    //     xinters.reserve(16);  // 2 is probably most common, but it's cheap to allocate 16
    //     for (int j=0; j<ny; j++) {
    //         double y = ygrid[j];
    //         if ((y < ymin) or (y > ymax)) {
    //             for (int i=0; i<nx; i++) {
    //                 out[j*nx+i] = false;
    //             }
    //             continue;
    //         }
    //         xinters.clear();
    //         // Loop through edges to find all relevant x intercepts
    //         double x1 = _xp[0];  // first point of segment
    //         double y1 = _yp[0];
    //         for (int k=0; k<_size; k++) {
    //             double x2 = _xp[k % _size];  // second point of segment
    //             double y2 = _yp[k % _size];
    //             if ((y > std::min(y1, y2)) && (y <= std::max(y1, y2)))
    //                 xinters.push_back((y-y1)*(x2-x1)/(y2-y1)+x1);
    //             x1 = x2;
    //             y1 = y2;
    //         }
    //         std::sort(xinters.begin(), xinters.end());
    //         // All points to the left of first intercept are outside the polygon
    //         // Alternate after that.
    //         bool contained = false;
    //         auto xptr = xinters.begin();
    //         for (int i=0; i<nx; i++) {
    //             if (xptr != xinters.end()) {
    //                 if (xgrid[i] > *xptr) {
    //                     contained = !contained;
    //                     xptr++;
    //                 }
    //             }
    //             out[j*ny+i] = contained;
    //         }
    //     }
    // }


    // #if defined(BATOID_GPU)
    //     #pragma omp end declare target
    // #endif

}
