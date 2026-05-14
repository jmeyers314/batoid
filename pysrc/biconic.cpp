#include "biconic.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportBiconic(py::module& m) {
        py::class_<Biconic, std::shared_ptr<Biconic>, Surface>(m, "CPPBiconic")
            .def(py::init<double, double, double, double>(), "init", 
                 "Rx"_a, "Ry"_a, "kx"_a, "ky"_a);
    }
}
