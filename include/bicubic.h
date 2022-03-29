#ifndef batoid_bicubic_h
#define batoid_bicubic_h

#include "surface.h"
#include "table.h"

namespace batoid {

    /////////////
    // Bicubic //
    /////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Bicubic : public Surface {
    public:
        Bicubic(const Table* table);
        ~Bicubic();

        virtual double sag(double, double) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;

    private:
        const Table* _table;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // BicubicHandle //
    ///////////////////

    class BicubicHandle : public SurfaceHandle {
    public:
        BicubicHandle(const TableHandle* handle);
        virtual ~BicubicHandle();
    };

}
#endif
