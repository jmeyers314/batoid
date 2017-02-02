#include <pybind11/pybind11.h>
#include "surface.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportSurface(py::module &m) {
        py::class_<Surface>(m, "Surface");
    }
}
