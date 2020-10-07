#ifndef batoid_coating_h
#define batoid_coating_h

namespace batoid {
    class Coating {
    public:
        Coating();
        virtual ~Coating();

        virtual void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double&transmit) const = 0;
        virtual double getReflect(double wavelength, double cosIncidenceAngle) const = 0;
        virtual double getTransmit(double wavelength, double cosIncidenceAngle) const = 0;

        virtual const Coating* getDevPtr() const = 0;

    protected:
        mutable Coating* _devPtr;
    };

    class SimpleCoating : public Coating {
    public:
        SimpleCoating(double reflectivity, double transmissivity);
        ~SimpleCoating();

        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;
        double getReflect(double wavelength, double cosIncidenceAngle) const override;
        double getTransmit(double wavelength, double cosIncidenceAngle) const override;

        virtual const Coating* getDevPtr() const override;

    private:
        double _reflectivity;
        double _transmissivity;
    };
}

#endif
