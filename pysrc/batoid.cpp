#include "batoid.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector(py::module&);

    void pyExportSurface(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportAsphere(py::module&);
    void pyExportBicubic(py::module&);
    void pyExportSphere(py::module&);
    void pyExportSum(py::module&);
    void pyExportParaboloid(py::module&);
    void pyExportPlane(py::module&);
    void pyExportPolynomialSurface(py::module&);

    void pyExportCoating(py::module&);
    void pyExportMedium(py::module&);
    void pyExportObscuration(py::module&);

    PYBIND11_MODULE(_batoid, m) {
        pyExportRayVector(m);

        pyExportSurface(m);
        pyExportQuadric(m);
        pyExportAsphere(m); // Order Surface, Quadric, Asphere important b/c inheritance
        pyExportBicubic(m);
        pyExportSphere(m);
        pyExportSum(m);
        pyExportParaboloid(m);
        pyExportPlane(m);
        pyExportPolynomialSurface(m);

        pyExportCoating(m);
        pyExportMedium(m);
        pyExportObscuration(m);

        using namespace pybind11::literals;

        m.def("applyForwardTransform", &applyForwardTransform);
        m.def("applyReverseTransform", &applyReverseTransform);
        m.def("intersect", &intersect);
        m.def("reflect", &reflect);
        m.def("refract", &refract);
        m.def("obscure", &obscure);
        m.def("rSplit", &rSplit);
    }
}
