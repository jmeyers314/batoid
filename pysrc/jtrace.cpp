#include "jtrace.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jtrace {
    void pyExportVec3(py::module&);
    void pyExportRay(py::module&);
    void pyExportIntersection(py::module&);
    void pyExportSurface(py::module&);
    void pyExportParaboloid(py::module&);
    void pyExportAsphere(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportSphere(py::module&);
    void pyExportPlane(py::module&);
    void pyExportTransformation(py::module&);
    void pyExportTable(py::module&);
    void pyExportMedium(py::module&);

    PYBIND11_PLUGIN(_jtrace) {
        py::module m("_jtrace", "ray tracer");
        pyExportVec3(m);
        pyExportRay(m);
        pyExportIntersection(m);
        pyExportSurface(m);
        pyExportParaboloid(m);
        pyExportAsphere(m);
        pyExportQuadric(m);
        pyExportSphere(m);
        pyExportPlane(m);
        pyExportTransformation(m);
        pyExportTable(m);
        pyExportMedium(m);

        return m.ptr();
    }
}
