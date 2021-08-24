#ifndef batoid_bicubic_h
#define batoid_bicubic_h

#include "surface.h"
#include "table.h"

namespace batoid {

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

}
#endif
