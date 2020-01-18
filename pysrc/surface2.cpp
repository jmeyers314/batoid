#include "surface2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface2(py::module& m) {
        py::class_<Surface2, std::shared_ptr<Surface2>>(m, "CPPSurface2")
            .def("intersectInPlace", &Surface2::intersectInPlace)
            .def("reflectInPlace", &Surface2::reflectInPlace)
            .def("refractInPlace", &Surface2::refractInPlace);
    }
}
