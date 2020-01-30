#include "batoid.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

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
    void pyExportCoating(py::module&);
    void pyExportMedium(py::module&);
    void pyExportObscuration(py::module&);
    void pyExportCoordSys(py::module&);
    void pyExportCoordTransform(py::module&);

    // GPU stuff
    void pyExportCoordTransform2(py::module&);
    void pyExportSurface2(py::module&);
    void pyExportRayVector2(py::module&);
    void pyExportMedium2(py::module&);
    void pyExportPlane2(py::module&);
    void pyExportSphere2(py::module&);
    void pyExportParaboloid2(py::module&);

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
        pyExportCoating(m);
        pyExportMedium(m);
        pyExportObscuration(m);
        pyExportCoordSys(m);
        pyExportCoordTransform(m);

        // GPU stuff
        pyExportCoordTransform2(m);
        pyExportSurface2(m);
        pyExportRayVector2(m);
        pyExportMedium2(m);
        pyExportPlane2(m);
        pyExportSphere2(m);
        pyExportParaboloid2(m);

        using namespace pybind11::literals;

        m.def("rayGrid",
             &rayGrid,
             "Make a RayVector in a grid",
             "zdist"_a, "length"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nside"_a, "wavelength"_a, "flux"_a, "medium"_a,
             "lattice"_a=false
         )
         .def("circularGrid",
             &circularGrid,
             "Make a RayVector on a circle",
             "zdist"_a, "outer"_a, "inner"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nradii"_a, "naz"_a, "wavelength"_a, "flux"_a, "medium"_a
         )
         .def("uniformCircularGrid",
             &uniformCircularGrid,
             "Make a RayVector with photon positions drawn uniformly from a circle",
             "zdist"_a, "outer"_a, "inner"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nrays"_a,
             "wavelength"_a, "flux"_a, "medium"_a, "seed"_a=0
         )
         .def("pointSourceCircularGrid",
             &pointSourceCircularGrid,
             "Make a spherically expanding RayVector from a point",
             "source"_a, "outer"_a, "inner"_a, "nradii"_a, "naz"_a, "wavelength"_a, "flux"_a, "medium"_a
         )
         .def("getNThread", &getNThread)
         .def("setNThread", &setNThread)
         .def("getMinChunk", &getMinChunk)
         .def("setMinChunk", &setMinChunk)
         .def("setRNGSeed", &setRNGSeed);

#if !((PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2))
        return m.ptr();
#endif
    }
}
