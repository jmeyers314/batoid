#include "sphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSphere(py::module& m) {
        py::class_<Sphere, std::shared_ptr<Sphere>, Surface>(m, "Sphere")
            .def(py::init<double,double,double,double>(), "init",
                 "R"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Sphere::getR)
            .def_property_readonly("B", &Sphere::getB)
            .def_property_readonly("Rin", &Sphere::getRin)
            .def_property_readonly("Rout", &Sphere::getRout);
    }
}
