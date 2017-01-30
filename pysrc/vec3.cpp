#include <pybind11/pybind11.h>
#include "vec3.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportVec3(py::module &m) {
        py::class_<Vec3>(m, "Vec3")
            .def(py::init<double,double,double>())
            .def(py::init<>())
            .def("MagnitudeSquared", &Vec3::MagnitudeSquared)
            .def("Magnitude", &Vec3::Magnitude)
            .def("UnitVec3", &Vec3::UnitVec3)
            .def("__repr__", &Vec3::repr);
    }
}
