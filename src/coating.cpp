#include "coating.h"
#include <new>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Coating::Coating() :
        _devPtr(nullptr)
    {}

    Coating::~Coating() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                freeDevPtr();
            }
        #endif
    }

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


    #if defined(BATOID_GPU)
    void Coating::freeDevPtr() const {
        if(_devPtr) {
            Coating* ptr = _devPtr;
            _devPtr = nullptr;
            #pragma omp target is_device_ptr(ptr)
            {
                delete ptr;
            }
        }
    }
    #endif

    const Coating* SimpleCoating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Coating* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new SimpleCoating(_reflectivity, _transmissivity);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }


}
