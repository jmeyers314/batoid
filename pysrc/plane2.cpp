#include "plane2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportPlane2(py::module& m) {
        py::class_<Plane2, std::shared_ptr<Plane2>, Surface2>(m, "CPPPlane2")
            .def(py::init<bool>(), "init", "allowReverse"_a=false);
    }
}
