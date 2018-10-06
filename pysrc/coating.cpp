#include "coating.h"
#include <pybind11/pybind11.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace batoid {
    void pyExportCoating(py::module& m) {
        py::class_<Coating, std::shared_ptr<Coating>>(m, "Coating")
            .def("getCoefs", &Coating::getCoefs);

        py::class_<SimpleCoating, std::shared_ptr<SimpleCoating>, Coating>(m, "SimpleCoating")
            .def(py::init<double,double>(), "init", "reflectivity"_a, "transmissivity"_a);
    }
}
