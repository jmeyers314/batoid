#ifndef batoid_table_h
#define batoid_table_h

#include <cstdlib>  // for size_t

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Table {
    public:
        Table(
            double x0, double y0, double dx, double dy,
            const double* z, const double* dzdx, const double* dzdy, const double* d2zdxdy,
            size_t nx, size_t ny
        );
        ~Table();

        const Table* getDevPtr() const;

        double eval(double, double) const;
        void grad(
            double x, double y,
            double& dzdx, double& dzdy
        ) const;

    protected:
        mutable Table* _devPtr;

    private:
        void freeDevPtr() const;

        const double _x0, _y0;
        const double _dx, _dy;
        const double* _z;
        const double* _dzdx;
        const double* _dzdy;
        const double* _d2zdxdy;
        const size_t _nx, _ny;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

}
#endif
