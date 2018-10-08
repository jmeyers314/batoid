#ifndef batoid_coating_h
#define batoid_coating_h

#include <string>

namespace batoid {
    class Coating {
    public:
        virtual ~Coating() {}

        virtual void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double&transmit) const = 0;

        virtual std::string repr() const = 0;
    };

    class SimpleCoating : public Coating {
    public:
        SimpleCoating(double reflectivity, double transmissivity);
        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;

        std::string repr() const override;

        double _reflectivity;  // public so pickle can see 'em
        double _transmissivity;
    };
}

#endif
