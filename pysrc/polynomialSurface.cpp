#include "polynomialSurface.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportPolynomialSurface(py::module& m) {
        py::class_<PolynomialSurface, std::shared_ptr<PolynomialSurface>, Surface>(m, "CPPPolynomialSurface")
            .def(py::init(
                [](
                    size_t coefs,
                    size_t coefs_gradx,
                    size_t coefs_grady,
                    size_t xsize,
                    size_t ysize
                ){
                    return new PolynomialSurface(
                        reinterpret_cast<const double*>(coefs),
                        reinterpret_cast<const double*>(coefs_gradx),
                        reinterpret_cast<const double*>(coefs_grady),
                        xsize, ysize
                    );
                }
            ));

        m.def(
            "horner2d",
            [](double x, double y, size_t coefs, size_t nx, size_t ny){
                return horner2d(
                    x, y,
                    reinterpret_cast<const double*>(coefs),
                    nx, ny
                );
            }
        );
    }
}
