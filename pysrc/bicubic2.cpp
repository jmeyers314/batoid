#include "bicubic2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBicubic2(py::module& m) {
        py::class_<Bicubic2, std::shared_ptr<Bicubic2>, Surface2>(m, "CPPBicubic2")
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
                    size_t size
                ){
                    return new Bicubic2(
                        x0, y0, dx, dy,
                        reinterpret_cast<double*>(z_ptr),
                        reinterpret_cast<double*>(dzdx_ptr),
                        reinterpret_cast<double*>(dzdy_ptr),
                        reinterpret_cast<double*>(d2zdxdy_ptr),
                        size
                    );
                }
            ));
    }
}
