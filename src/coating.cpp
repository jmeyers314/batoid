#include "coating.h"
#include <new>
#include <omp.h>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    /////////////
    // Coating //
    /////////////

    Coating::Coating() {}

    Coating::~Coating() {}


    ///////////////////
    // SimpleCoating //
    ///////////////////

    SimpleCoating::SimpleCoating(double reflectivity, double transmissivity) :
        _reflectivity(reflectivity), _transmissivity(transmissivity)
    {}

    SimpleCoating::~SimpleCoating() {}

    void SimpleCoating::getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const {
        reflect = _reflectivity;
        transmit = _transmissivity;
    }

    double SimpleCoating::getReflect(double wavelength, double cosIncidenceAngle) const {
        return _reflectivity;
    }

    double SimpleCoating::getTransmit(double wavelength, double cosIncidenceAngle) const {
        return _transmissivity;
    }


    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // CoatingHandle //
    ///////////////////

    CoatingHandle::CoatingHandle() :
        _hostPtr(nullptr),
        _devicePtr(nullptr)
    {}

    CoatingHandle::~CoatingHandle() {}

    const Coating* CoatingHandle::getPtr() const {
        #if defined(BATOID_GPU)
            return _devicePtr;
        #else
            return _hostPtr;
        #endif
    }

    const Coating* CoatingHandle::getHostPtr() const {
        return _hostPtr;
    }


    /////////////////////////
    // SimpleCoatingHandle //
    /////////////////////////

    SimpleCoatingHandle::SimpleCoatingHandle(double reflectivity, double transmissivity) :
        CoatingHandle()
    {
        _hostPtr = new SimpleCoating(reflectivity, transmissivity);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(SimpleCoating), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) SimpleCoating(reflectivity, transmissivity);
            }
        #endif
    }

    SimpleCoatingHandle::~SimpleCoatingHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<SimpleCoating *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~SimpleCoating();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }

}
