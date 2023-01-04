#include <new>
#include "bicubic.h"


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Bicubic::Bicubic(
        const Table* table
    ) :
        Surface(), _table(table)
    {}

    Bicubic::~Bicubic() {}  // don't own _table

    double Bicubic::sag(double x, double y) const {
        return _table->eval(x, y);
    }

    void Bicubic::normal(
        double x, double y,
        double& nx, double& ny, double& nz
    ) const {
        double dxdz, dydz;
        _table->grad(x, y, dxdz, dydz);
        if (std::isnan(dxdz)) {
            nx = NAN;
            ny = NAN;
            nz = NAN;
            return;
        }

        // This works
        nz = 1/std::sqrt(1 + dxdz*dxdz + dydz*dydz);
        nx = -dxdz*nz;
        ny = -dydz*nz;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Surface* Bicubic::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                const Table* tableDevPtr = _table->getDevPtr();

                #pragma omp target map(from:ptr) is_device_ptr(tableDevPtr)
                {
                    ptr = new Bicubic(tableDevPtr);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }
}
