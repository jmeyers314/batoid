#include "batoid.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#if defined(BATOID_GPU)
    #include "omp.h"
#endif

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector(py::module&);

    void pyExportTable(py::module&);

    void pyExportSurface(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportAsphere(py::module&);
    void pyExportTilted(py::module&);
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

        pyExportTable(m);

        pyExportSurface(m);
        pyExportQuadric(m);
        pyExportAsphere(m); // Order Surface, Quadric, Asphere important b/c inheritance
        pyExportTilted(m);
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
        m.def("refractScreen", &refractScreen);
        m.def("obscure", &obscure);
        m.def("rSplit", &rSplit);
        m.def(
            "applyForwardTransformArrays",
            [](
                const vec3 dr,
                const mat3 drot,
                size_t x,
                size_t y,
                size_t z,
                size_t n
            ){
                applyForwardTransformArrays(
                    dr, drot,
                    reinterpret_cast<double*>(x),
                    reinterpret_cast<double*>(y),
                    reinterpret_cast<double*>(z),
                    n
                );
            }
        );
        m.def(
            "applyReverseTransformArrays",
            [](
                const vec3 dr,
                const mat3 drot,
                size_t x,
                size_t y,
                size_t z,
                size_t n
            ){
                applyReverseTransformArrays(
                    dr, drot,
                    reinterpret_cast<double*>(x),
                    reinterpret_cast<double*>(y),
                    reinterpret_cast<double*>(z),
                    n
                );
            }
        );
        m.def(
            "finishParallel",
            [](
                const vec3 dr,
                const mat3 drot,
                const vec3 vv,
                size_t r,
                size_t n
            ){
                finishParallel(
                    dr, drot, vv,
                    reinterpret_cast<double*>(r),
                    n
                );
            });
        #if defined(BATOID_GPU)
            m.def(
                "get_omp_environment",
                []() {
                    py::dict out;
                    out.attr("__setitem__")("initial_device", omp_get_initial_device());
                    out.attr("__setitem__")("default_device", omp_get_default_device());
                    out.attr("__setitem__")("num_devices", omp_get_num_devices());
                    out.attr("__setitem__")("device_num", omp_get_device_num());
                    return out;
                }
            );
        #endif
    }
}
