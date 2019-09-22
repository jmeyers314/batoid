#include "coating.h"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

namespace py = pybind11;
using namespace pybind11::literals;


namespace batoid {
    void pyExportCoating(py::module& m) {
        py::class_<Coating, std::shared_ptr<Coating>>(m, "CPPCoating")
            .def("getCoefs", [](const Coating& coating, double wavelength, double cosIncidenceAngle){
                double reflect, transmit;
                coating.getCoefs(wavelength, cosIncidenceAngle, reflect, transmit);
                return py::make_tuple(reflect, transmit);
            })
            .def("getReflect", &Coating::getReflect)
            .def("getTransmit", &Coating::getTransmit)
            .def("__repr__", &Coating::repr);

        py::class_<SimpleCoating, std::shared_ptr<SimpleCoating>, Coating>(m, "CPPSimpleCoating")
            .def(py::init<double,double>(), "init", "reflectivity"_a, "transmissivity"_a)
            .def_readonly("reflectivity", &SimpleCoating::_reflectivity)
            .def_readonly("transmissivity", &SimpleCoating::_transmissivity)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const SimpleCoating& sc){
                    return py::make_tuple(sc._reflectivity, sc._transmissivity);
                },
                [](py::tuple t) {
                    return SimpleCoating(t[0].cast<double>(), t[1].cast<double>());
                }
            ))
            .def("__hash__", [](const SimpleCoating& sc) {
                return py::hash(py::make_tuple(
                    "CPPSimpleCoating",
                    sc._reflectivity,
                    sc._transmissivity
                ));
            });
    }
}
