#include "bicubic.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBicubic(py::module& m) {
        py::class_<BicubicHandle, std::shared_ptr<BicubicHandle>, SurfaceHandle>(m, "CPPBicubic")
            .def(py::init<const TableHandle*>(), "init", "table"_a);
    }
}
