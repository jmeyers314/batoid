#include "coating.h"
#include <sstream>

namespace batoid {
    SimpleCoating::SimpleCoating(double reflectivity, double transmissivity) :
        _reflectivity(reflectivity), _transmissivity(transmissivity) {}

    void SimpleCoating::getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const {
        reflect = _reflectivity;
        transmit = _transmissivity;
    }

    std::string SimpleCoating::repr() const {
        std::ostringstream oss;
        oss << "SimpleCoating("
            << _reflectivity
            << ", "
            << _transmissivity
            << ")";
        return oss.str();
    }

}
