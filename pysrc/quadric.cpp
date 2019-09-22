#include "quadric.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportQuadric(py::module& m) {
        py::class_<Quadric, std::shared_ptr<Quadric>, Surface>(m, "CPPQuadric")
            .def(py::init<double,double>(), "init", "R"_a, "conic"_a)
            .def_property_readonly("R", &Quadric::getR)
            .def_property_readonly("conic", &Quadric::getConic);
    }
}
