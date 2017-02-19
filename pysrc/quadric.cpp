#include <memory>
#include <pybind11/pybind11.h>
#include "quadric.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportQuadric(py::module& m) {
        py::class_<Quadric, std::shared_ptr<Quadric>, Surface>(m, "Quadric")
            .def(py::init<double,double,double>())
            .def_property_readonly("R", &Quadric::getR)
            .def_property_readonly("kappa", &Quadric::getKappa)
            .def_property_readonly("B", &Quadric::getB)
            .def("__call__", &Quadric::operator())
            .def("normal", &Quadric::normal)
            .def("intersect", &Quadric::intersect)
            .def("__repr__", &Quadric::repr);
    }
}
