#include "sphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSphere(py::module& m) {
        py::class_<Sphere, std::shared_ptr<Sphere>, Surface>(m, "Sphere")
            .def(py::init<double>(), "init", "R"_a)
            .def_property_readonly("R", &Sphere::getR)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const Sphere& s) { return py::make_tuple(s.getR()); },
                [](py::tuple t) { return Sphere(t[0].cast<double>()); }
            ))
            .def("__hash__", [](const Sphere& s) {
                return py::hash(py::make_tuple("Sphere", s.getR()));
            });
    }
}
