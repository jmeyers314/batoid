#include "batoid.h"
#include <pybind11/pybind11.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportVec3(py::module&);
    void pyExportVec2(py::module&);
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
    void pyExportObscuration(py::module&);
    void pyExportXForm(py::module&);
    void pyExportCoordSys(py::module&);
    void pyExportCoordTransform(py::module&);

#if (PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2)
    PYBIND11_MODULE(_batoid, m) {
#else
    PYBIND11_PLUGIN(_batoid) {
        py::module m("_batoid", "ray tracer");
#endif
        pyExportVec3(m);
        pyExportVec2(m);
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
        pyExportObscuration(m);
        pyExportXForm(m);
        pyExportCoordSys(m);
        pyExportCoordTransform(m);

        m.def("reflect", (Ray (*)(const Ray&, const Surface&)) &reflect)
         .def("reflect", (std::vector<Ray> (*)(const std::vector<Ray>&, const Surface&)) &reflect)
         .def("refract", (Ray (*)(const Ray&, const Surface&, const double, const double)) &refract)
         .def("refract", (std::vector<Ray> (*)(const std::vector<Ray>&, const Surface&, const double, const double)) &refract)
         .def("refract", (Ray (*)(const Ray&, const Surface&, const Medium&, const Medium&)) &refract)
         .def("refract", (std::vector<Ray> (*)(const std::vector<Ray>&, const Surface&, const Medium&, const Medium&)) &refract)
         .def("rayGrid", (std::vector<Ray> (*)(double, double, double, double, int, double, double)) &rayGrid)
         .def("rayGrid", (std::vector<Ray> (*)(double, double, double, double, int, double, const Medium&)) &rayGrid)
         .def("circularGrid", (std::vector<Ray> (*)(double, double, double, double, double, int, int, double, double)) &circularGrid)
         .def("circularGrid", (std::vector<Ray> (*)(double, double, double, double, double, int, int, double, const Medium&)) &circularGrid)
         .def("trimVignetted", &trimVignetted)
         .def("trimVignettedInPlace", &trimVignettedInPlace);

#if !((PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2))
        return m.ptr();
#endif
    }
}
