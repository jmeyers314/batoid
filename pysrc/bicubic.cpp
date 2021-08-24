#include "bicubic.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBicubic(py::module& m) {
        py::class_<Bicubic, std::shared_ptr<Bicubic>, Surface>(m, "CPPBicubic")
            .def(py::init<const Table*>(), "init", "table"_a);
    }
}
