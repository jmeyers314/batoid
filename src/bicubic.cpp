#include <new>
#include "bicubic.h"
#include <omp.h>


namespace batoid {

    /////////////
    // Bicubic //
    /////////////

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

        nz = 1/std::sqrt(1 + dxdz*dxdz + dydz*dydz);
        nx = -dxdz*nz;
        ny = -dydz*nz;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // BicubicHandle //
    ///////////////////

    BicubicHandle::BicubicHandle(const TableHandle* handle) :
        SurfaceHandle()
    {
        _hostPtr = new Bicubic(handle->getHostPtr());
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Bicubic), omp_get_default_device());
            const Table* table = handle->getPtr();
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc, table)
            {
                _devicePtr = new (alloc) Bicubic(table);
            }
        #endif
    }

    BicubicHandle::~BicubicHandle() {
        #if defined(BATOID_GPU)
            // We know following is noop, but compiler might not...

            // auto devPtr = static_cast<Bicubic *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Bicubic();
            // }
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }
}
