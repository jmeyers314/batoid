#include "paraboloid.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportParaboloid(py::module& m) {
        py::class_<Paraboloid, std::shared_ptr<Paraboloid>, Surface>(m, "Paraboloid")
            .def(py::init<double>(), "init", "R"_a)
            .def_property_readonly("R", &Paraboloid::getR)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const Paraboloid& p) { return py::make_tuple(p.getR()); },
                [](py::tuple t) { return Paraboloid(t[0].cast<double>()); }
            ))
            .def("__hash__", [](const Paraboloid& p) {
                return py::hash(py::make_tuple("Paraboloid", p.getR()));
            });
    }
}
