#include <pybind11/pybind11.h>
#include "surface.h"
#include "transformation.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportSurface(py::module &m) {
        py::class_<Surface>(m, "Surface")
            .def("shift", &Surface::shift)
            .def("rotX", &Surface::rotX)
            .def("rotY", &Surface::rotY)
            .def("rotZ", &Surface::rotZ);
    }
}
