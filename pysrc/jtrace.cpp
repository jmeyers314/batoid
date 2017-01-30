#include <pybind11/pybind11.h>
#include "jtrace.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportVec3(py::module &);
    void pyExportRay(py::module &);
    void pyExportIntersection(py::module &);
    void pyExportSurface(py::module &);
    void pyExportParaboloid(py::module &);

    PYBIND11_PLUGIN(jtrace) {
        py::module m("jtrace", "ray tracer");

        pyExportVec3(m);
        pyExportRay(m);
        pyExportIntersection(m);
        pyExportSurface(m);
        pyExportParaboloid(m);

        return m.ptr();
    }
}
