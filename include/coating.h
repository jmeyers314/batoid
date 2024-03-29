#ifndef batoid_coating_h
#define batoid_coating_h

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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

    private:
        #if defined(BATOID_GPU)
        void freeDevPtr() const;
        #endif
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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

}

#endif
