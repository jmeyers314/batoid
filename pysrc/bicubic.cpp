#include "bicubic.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBicubic(py::module& m) {
        py::class_<Bicubic, std::shared_ptr<Bicubic>, Surface>(m, "CPPBicubic")
            .def(py::init(
                [](
                    double x0,
                    double y0,
                    double dx,
                    double dy,
                    size_t z_ptr,
                    size_t dzdx_ptr,
                    size_t dzdy_ptr,
                    size_t d2zdxdy_ptr,
                    size_t nx,
                    size_t ny
                ){
                    return new Bicubic(
                        x0, y0, dx, dy,
                        reinterpret_cast<double*>(z_ptr),
                        reinterpret_cast<double*>(dzdx_ptr),
                        reinterpret_cast<double*>(dzdy_ptr),
                        reinterpret_cast<double*>(d2zdxdy_ptr),
                        nx, ny
                    );
                }
            ));
    }
}
