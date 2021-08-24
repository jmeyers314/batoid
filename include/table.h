#ifndef batoid_table_h
#define batoid_table_h

#include <cstdlib>  // for size_t

namespace batoid {

    class Table {
    public:
        Table(
            double x0, double y0, double dx, double dy,
            const double* z, const double* dzdx, const double* dzdy, const double*d2zdxdy,
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
        const double _x0, _y0;
        const double _dx, _dy;
        const double* _z;
        const double* _dzdx;
        const double* _dzdy;
        const double* _d2zdxdy;
        const size_t _nx, _ny;
    };

}
#endif
