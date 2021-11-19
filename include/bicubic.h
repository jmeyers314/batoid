#ifndef batoid_bicubic_h
#define batoid_bicubic_h

#include "surface.h"
#include "table.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Bicubic : public Surface {
    public:
        Bicubic(const Table* table);
        ~Bicubic();

        virtual const Surface* getDevPtr() const override;

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

}
#endif
