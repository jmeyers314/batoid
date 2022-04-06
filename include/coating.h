#ifndef batoid_coating_h
#define batoid_coating_h

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    /////////////
    // Coating //
    /////////////

    class Coating {
    public:
        Coating();
        virtual ~Coating();

        virtual void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double&transmit) const = 0;
        virtual double getReflect(double wavelength, double cosIncidenceAngle) const = 0;
        virtual double getTransmit(double wavelength, double cosIncidenceAngle) const = 0;
    };


    ///////////////////
    // SimpleCoating //
    ///////////////////

    class SimpleCoating : public Coating {
    public:
        SimpleCoating(double reflectivity, double transmissivity);
        ~SimpleCoating();

        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;
        double getReflect(double wavelength, double cosIncidenceAngle) const override;
        double getTransmit(double wavelength, double cosIncidenceAngle) const override;

    private:
        double _reflectivity;
        double _transmissivity;
    };


    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // CoatingHandle //
    ///////////////////

    class CoatingHandle {
    public:
        CoatingHandle();

        virtual ~CoatingHandle();

        const Coating* getPtr() const;

        const Coating* getHostPtr() const;

    protected:
        Coating* _hostPtr;
        Coating* _devicePtr;
    };


    /////////////////////////
    // SimpleCoatingHandle //
    /////////////////////////

    class SimpleCoatingHandle : public CoatingHandle {
    public:
        SimpleCoatingHandle(double reflectivity, double transmissivity);
        virtual ~SimpleCoatingHandle();
    };

}

#endif
