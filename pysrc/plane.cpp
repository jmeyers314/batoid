#include "plane.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportPlane(py::module& m) {
        py::class_<Plane, std::shared_ptr<Plane>, Surface>(m, "CPPPlane")
            .def(py::init<>());
    }
}
