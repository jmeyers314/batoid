#ifndef batoid_table_h
#define batoid_table_h

#include <cstdlib>  // for size_t

namespace batoid {

    ///////////
    // Table //
    ///////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Table {
    public:
        Table(
            double x0, double y0, double dx, double dy,
            const double* z,
            const double* dzdx,
            const double* dzdy,
            const double* d2zdxdy,
            size_t nx, size_t ny
        );
        ~Table();

        double eval(double, double) const;
        void grad(
            double x, double y,
            double& dzdx, double& dzdy
        ) const;

    private:
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


    /////////////////
    // TableHandle //
    /////////////////

    class TableHandle {
    public:
        TableHandle(
            double x0, double y0, double dx, double dy,
            const double* z,
            const double* dzdx,
            const double* dzdy,
            const double* d2zdxdy,
            size_t nx, size_t ny
        );

        ~TableHandle();

        const Table* getPtr() const;

        const Table* getHostPtr() const;

    private:
        const double* _z;
        const double* _dzdx;
        const double* _dzdy;
        const double* _d2zdxdy;
        const size_t _size;
        Table* _hostPtr;
        Table* _devicePtr;
    };

}
#endif
