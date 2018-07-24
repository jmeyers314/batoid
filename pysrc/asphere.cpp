#include "asphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportAsphere(py::module& m) {
        py::class_<Asphere, std::shared_ptr<Asphere>, Surface>(m, "Asphere")
            .def(py::init<double,double,std::vector<double>>(), "init", "R"_a, "conic"_a, "coefs"_a)
            .def_property_readonly("R", &Asphere::getR)
            .def_property_readonly("conic", &Asphere::getConic)
            .def_property_readonly("coefs", &Asphere::getCoefs)
            .def(py::pickle(
                [](const Asphere& a) {
                    return py::make_tuple(a.getR(), a.getConic(), a.getCoefs());
                },
                [](py::tuple t) {
                    return Asphere(
                        t[0].cast<double>(),
                        t[1].cast<double>(),
                        t[2].cast<std::vector<double>>()
                    );
                }
            ))
            .def("__hash__", [](const Asphere& a) {
                return py::hash(py::make_tuple(
                    "Asphere",
                    a.getR(),
                    a.getConic(),
                    py::tuple(py::cast(a.getCoefs()))
                ));
            });
    }
}
