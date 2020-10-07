#include "coating.h"
#include <new>

namespace batoid {

    #if defined _OPENMP && _OPENMP >= 201511
        #pragma omp declare target
    #endif

    Coating::Coating() :
        _devPtr(nullptr)
    {}

    Coating::~Coating() {}


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

    #if defined _OPENMP && _OPENMP >= 201511
        #pragma omp end declare target
    #endif


    const Coating* SimpleCoating::getDevPtr() const {
        #if defined _OPENMP && _OPENMP >= 201511
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
