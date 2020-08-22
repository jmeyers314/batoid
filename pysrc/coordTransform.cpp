#include "coordTransform.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    void pyExportCoordTransform(py::module& m) {
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CPPCoordTransform")
            .def(py::init<const CoordSys&, const CoordSys&>())
            .def("applyForwardInPlace", &CoordTransform::applyForwardInPlace)
            .def("applyReverseInPlace", &CoordTransform::applyReverseInPlace)
            .def_readonly("source", &CoordTransform::source)
            .def_readonly("destination", &CoordTransform::destination)
            .def_readonly("dr", &CoordTransform::dr)
            .def_readonly("rot", &CoordTransform::rot);
    }
}
