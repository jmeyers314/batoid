#include "coating.h"
#include <sstream>

namespace batoid {
    SimpleCoating::SimpleCoating(double reflectivity, double transmissivity) :
        _reflectivity(reflectivity), _transmissivity(transmissivity) {}

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

    bool SimpleCoating::operator==(const Coating& rhs) const {
        if (const SimpleCoating* other = dynamic_cast<const SimpleCoating*>(&rhs)) {
            return _reflectivity == other->_reflectivity &&
                   _transmissivity == other->_transmissivity;
        } else return false;
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
