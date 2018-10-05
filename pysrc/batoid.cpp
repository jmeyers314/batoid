#include "batoid.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRay(py::module&);
    void pyExportRayVector(py::module&);

    void pyExportSurface(py::module&);
    void pyExportAsphere(py::module&);
    void pyExportBicubic(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportSphere(py::module&);
    void pyExportSum(py::module&);
    void pyExportParaboloid(py::module&);
    void pyExportPlane(py::module&);
    void pyExportPolynomialSurface(py::module&);

    void pyExportTable(py::module&);
    void pyExportMedium(py::module&);
    void pyExportObscuration(py::module&);
    void pyExportCoordSys(py::module&);
    void pyExportCoordTransform(py::module&);

#if (PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2)
    PYBIND11_MODULE(_batoid, m) {
#else
    PYBIND11_PLUGIN(_batoid) {
        py::module m("_batoid", "ray tracer");
#endif
        pyExportRay(m);
        pyExportRayVector(m);

        pyExportSurface(m);
        pyExportAsphere(m);
        pyExportBicubic(m);
        pyExportQuadric(m);
        pyExportSphere(m);
        pyExportParaboloid(m);
        pyExportPlane(m);
        pyExportPolynomialSurface(m);
        pyExportSum(m);

        pyExportTable(m);
        pyExportMedium(m);
        pyExportObscuration(m);
        pyExportCoordSys(m);
        pyExportCoordTransform(m);

        using namespace pybind11::literals;

        m.def("rayGrid",
             &rayGrid,
             "Make a RayVector in a grid",
             "zdist"_a, "length"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nside"_a, "wavelength"_a, "flux"_a, "medium"_a
         )
         .def("circularGrid",
             &circularGrid,
             "Make a RayVector on a circle",
             "zdist"_a, "outer"_a, "inner"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nradii"_a, "naz"_a, "wavelength"_a, "flux"_a, "medium"_a
         )
         .def("getNThread", &getNThread)
         .def("setNThread", &setNThread);

#if !((PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2))
        return m.ptr();
#endif
    }
}
