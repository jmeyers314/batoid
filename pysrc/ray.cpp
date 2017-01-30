#include <pybind11/pybind11.h>
#include "ray.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportRay(py::module &m) {
        py::class_<Ray>(m, "Ray")
            .def(py::init<double,double,double,double,double,double,double>(),
                 "init",
                 "x0"_a, "y0"_a, "z0"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0)
            .def(py::init<Vec3,Vec3,double>(),
                 "init",
                 "p0"_a, "v"_a, "t"_a=0.0)
            .def_readonly("p0", &Ray::p0)
            .def_readonly("v", &Ray::v)
            .def_readonly("t0", &Ray::t0)
            .def("__call__", &Ray::operator())
            .def("__repr__", &Ray::repr);
    }
}
