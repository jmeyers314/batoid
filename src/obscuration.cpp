#include "obscuration.h"
#include <new>
#include <cmath>
#include <algorithm>
#include <vector>


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscCircle::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Obscuration* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscCircle(_radius, _x0, _y0);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        ObscAnnulus::ObscAnnulus(double inner, double outer, double x0, double y0) :
            _inner(inner), _outer(outer), _x0(x0), _y0(y0)
        {}

        ObscAnnulus::~ObscAnnulus() {}

        bool ObscAnnulus::contains(double x, double y) const {
            double h = std::hypot(x-_x0, y-_y0);
            return (_inner <= h) && (h < _outer);
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscAnnulus::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Obscuration* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscAnnulus(_inner, _outer, _x0, _y0);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscRectangle::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            ObscRectangle* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscRectangle(_width, _height, _x0, _y0, _theta);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscRay::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            ObscRay* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscRay(_width, _theta, _x0, _y0);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        ObscPolygon::ObscPolygon(const double* xp, const double* yp, const size_t size) :
            Obscuration(), _xp(xp), _yp(yp), _size(size)
        {}

        ObscPolygon::~ObscPolygon() {
            #if defined(BATOID_GPU)
                if (_devPtr) {
                    Obscuration* ptr = _devPtr;
                    #pragma omp target is_device_ptr(ptr)
                    {
                        delete ptr;
                    }

                    const double* xp = _xp;
                    const double* yp = _yp;
                    #pragma omp target exit data \
                        map(release:xp[:_size], yp[:_size])
                }
            #endif
        }

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    // CPU only for now.
    void ObscPolygon::containsGrid(
        const double* xgrid, const double* ygrid, bool* out, const size_t nx, const size_t ny
    ) const {
        // xgrid is [nx], ygrid is [ny]
        // out is [ny, nx]

        // Compute polygon y min/max
        double ymin = _yp[0];
        double ymax = _yp[0];
        for (int k=1; k<_size; k++) {
            if (_yp[k] < ymin)
                ymin = _yp[k];
            if (_yp[k] > ymax)
                ymax = _yp[k];
        }

        // Proceed row by row
        std::vector<double> xinters;
        xinters.reserve(16);  // 2 is probably most common, but it's cheap to allocate 16
        for (int j=0; j<ny; j++) {
            double y = ygrid[j];
            if ((y < ymin) or (y > ymax)) {
                for (int i=0; i<nx; i++) {
                    out[j*nx+i] = false;
                }
                continue;
            }
            xinters.clear();
            // Loop through edges to find all relevant x intercepts
            double x1 = _xp[0];  // first point of segment
            double y1 = _yp[0];
            for (int k=0; k<_size; k++) {
                double x2 = _xp[k % _size];  // second point of segment
                double y2 = _yp[k % _size];
                if ((y > std::min(y1, y2)) && (y <= std::max(y1, y2)))
                    xinters.push_back((y-y1)*(x2-x1)/(y2-y1)+x1);
                x1 = x2;
                y1 = y2;
            }
            std::sort(xinters.begin(), xinters.end());
            // All points to the left of first intercept are outside the polygon
            // Alternate after that.
            bool contained = false;
            auto xptr = xinters.begin();
            for (int i=0; i<nx; i++) {
                if (xptr != xinters.end()) {
                    if (xgrid[i] > *xptr) {
                        contained = !contained;
                        xptr++;
                    }
                }
                out[j*ny+i] = contained;
            }
        }
    }

    const Obscuration* ObscPolygon::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Obscuration* ptr;
                // Allocate arrays on device
                const double* xp = _xp;
                const double* yp = _yp;
                #pragma omp target enter data map(to:xp[:_size], yp[:_size])
                #pragma omp target map(from:ptr)
                {
                    ptr = new ObscPolygon(xp, yp, _size);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        ObscNegation::ObscNegation(const Obscuration* original) :
            Obscuration(), _original(original)
        {}

        ObscNegation::~ObscNegation() {}

        bool ObscNegation::contains(double x, double y) const {
            return !_original->contains(x, y);
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscNegation::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            ObscNegation* ptr;
            const Obscuration* originalDevPtr = _original->getDevPtr();

            #pragma omp target map(from:ptr) is_device_ptr(originalDevPtr)
            {
                ptr = new ObscNegation(originalDevPtr);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        ObscUnion::ObscUnion(const Obscuration** obscs, size_t nobsc) :
            Obscuration(), _obscs(obscs), _nobsc(nobsc)
        {}

        ObscUnion::~ObscUnion() {
            #if defined(BATOID_GPU)
                if (_devPtr) {
                    const Obscuration** obscs = _obscs;
                    #pragma omp target exit data map(release:obscs[:_nobsc])
                }
            #endif
        }

        bool ObscUnion::contains(double x, double y) const {
            bool ret = false;
            for (int i=0; i<_nobsc; i++) {
                ret |= _obscs[i]->contains(x, y);
            }
            return ret;
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscUnion::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            const Obscuration** obscs = new const Obscuration*[_nobsc];
            for (int i=0; i<_nobsc; i++) {
                obscs[i] = _obscs[i]->getDevPtr();
            }
            Obscuration* ptr;
            #pragma omp target enter data map(to:obscs[:_nobsc])
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscUnion(obscs, _nobsc);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        ObscIntersection::ObscIntersection(const Obscuration** obscs, size_t nobsc) :
            Obscuration(), _obscs(obscs), _nobsc(nobsc)
        {}

        ObscIntersection::~ObscIntersection() {
            #if defined(BATOID_GPU)
                if (_devPtr) {
                    const Obscuration** obscs = _obscs;
                    #pragma omp target exit data map(release:obscs[:_nobsc])
                }
            #endif
        }

        bool ObscIntersection::contains(double x, double y) const {
            bool ret = true;
            for (int i=0; i<_nobsc; i++) {
                ret &= _obscs[i]->contains(x, y);
            }
            return ret;
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Obscuration* ObscIntersection::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            const Obscuration** obscs = new const Obscuration*[_nobsc];
            for (int i=0; i<_nobsc; i++) {
                obscs[i] = _obscs[i]->getDevPtr();
            }
            Obscuration* ptr;
            #pragma omp target enter data map(to:obscs[:_nobsc])
            #pragma omp target map(from:ptr)
            {
                ptr = new ObscIntersection(obscs, _nobsc);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }
}
