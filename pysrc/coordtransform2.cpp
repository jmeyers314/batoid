#include "coordtransform2.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace batoid {
    void pyExportCoordTransform2(py::module& m) {
        py::class_<CoordTransform2, std::shared_ptr<CoordTransform2>>(m, "CPPCoordTransform2")
            .def(py::init<const CoordSys&, const CoordSys&>())
            .def("applyForwardInPlace", (void (CoordTransform2::*)(RayVector2&) const) &CoordTransform2::applyForwardInPlace)
            .def("applyReverseInPlace", (void (CoordTransform2::*)(RayVector2&) const) &CoordTransform2::applyReverseInPlace);
    }
}
