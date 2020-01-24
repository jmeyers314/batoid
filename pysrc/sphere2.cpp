#include "sphere2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSphere2(py::module& m) {
        py::class_<Sphere2, std::shared_ptr<Sphere2>, Surface2>(m, "CPPSphere2")
            .def(py::init<double>(), "init");
    }
}
