#include <memory>
#include <pybind11/pybind11.h>
#include "surface.h"
#include "transformation.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("shift", (Transformation (Surface::*)(double,double,double) const) &Surface::shift)
            .def("shift", (Transformation (Surface::*)(const Vec3&) const) &Surface::shift)
            .def("rotX", &Surface::rotX)
            .def("rotY", &Surface::rotY)
            .def("rotZ", &Surface::rotZ);
    }
}
