#ifndef batoid_coating_h
#define batoid_coating_h

namespace batoid {
    class Coating {
    public:
        virtual void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double&transmit) const = 0;
    };

    class SimpleCoating : public Coating {
    public:
        SimpleCoating(double reflectivity, double transmissivity);
        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;

    private:
        double _reflectivity;
        double _transmissivity;
    };
}

#endif
