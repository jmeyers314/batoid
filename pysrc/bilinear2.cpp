#include "bilinear2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBilinear2(py::module& m) {
        py::class_<Bilinear2, std::shared_ptr<Bilinear2>, Surface2>(m, "CPPBilinear2")
            .def(py::init(
                [](
                    double x0, double y0,
                    double dx, double dy,
                    size_t z_ptr,
                    size_t size
                ){
                    return new Bilinear2(
                        x0, y0, dx, dy,
                        reinterpret_cast<double*>(z_ptr),
                        size
                    );
                }
            ));
    }
}
