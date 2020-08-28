#include "batoid.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector(py::module&);

    void pyExportSurface(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportAsphere(py::module&);
    // void pyExportBicubic(py::module&);
    void pyExportSphere(py::module&);
    // void pyExportSum(py::module&);
    void pyExportParaboloid(py::module&);
    void pyExportPlane(py::module&);
    // void pyExportPolynomialSurface(py::module&);

    // void pyExportTable(py::module&);
    // void pyExportCoating(py::module&);
    void pyExportMedium(py::module&);
    // void pyExportObscuration(py::module&);

    PYBIND11_MODULE(_batoid, m) {
        pyExportRayVector(m);

        pyExportSurface(m);
        pyExportQuadric(m);
        pyExportAsphere(m); // Order Surface, Quadric, Asphere important b/c inheritance
        // pyExportBicubic(m);
        pyExportSphere(m);
        // pyExportSum(m);
        pyExportParaboloid(m);
        pyExportPlane(m);
        // pyExportPolynomialSurface(m);

        // pyExportTable(m);
        // pyExportCoating(m);
        pyExportMedium(m);
        // pyExportObscuration(m);

        using namespace pybind11::literals;

        m.def("intersect", &intersect);
        m.def("applyForwardTransform", &applyForwardTransform);
        m.def("applyReverseTransform", &applyReverseTransform);

        //  .def("getNThread", &getNThread)
        //  .def("setNThread", &setNThread)
        //  .def("getMinChunk", &getMinChunk)
        //  .def("setMinChunk", &setMinChunk)
        //  .def("setRNGSeed", &setRNGSeed);
    }
}
