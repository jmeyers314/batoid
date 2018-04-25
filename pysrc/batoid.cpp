#include "batoid.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRay(py::module&);
    void pyExportSurface(py::module&);
    void pyExportParaboloid(py::module&);
    void pyExportAsphere(py::module&);
    void pyExportQuadric(py::module&);
    void pyExportSphere(py::module&);
    void pyExportPlane(py::module&);
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
        pyExportSurface(m);
        pyExportParaboloid(m);
        pyExportAsphere(m);
        pyExportQuadric(m);
        pyExportSphere(m);
        pyExportPlane(m);
        pyExportTable(m);
        pyExportMedium(m);
        pyExportObscuration(m);
        pyExportCoordSys(m);
        pyExportCoordTransform(m);

        using namespace pybind11::literals;

        m.def("reflect", (Ray (*)(const Ray&, const Surface&)) &reflect)
         .def("reflect", [](const RayVector& rv, const Surface& s){
             RayVector result;
             result.rays = std::move(reflect(rv.rays, s));
             return result;
         })
         .def("reflectInPlace", (void (*)(Ray&, const Surface&)) &reflectInPlace)
         .def("reflectInPlace", [](RayVector& rv, const Surface& s){
             reflectInPlace(rv.rays, s);
         })
         .def("refract", (Ray (*)(const Ray&, const Surface&, const double, const double)) &refract)
         .def("refract", [](const RayVector& rv, const Surface& s, const double n1, const double n2){
             RayVector result;
             result.rays = std::move(refract(rv.rays, s, n1, n2));
             return result;
         })
         .def("refract", (Ray (*)(const Ray&, const Surface&, const Medium&, const Medium&)) &refract)
         .def("refract", [](const RayVector& rv, const Surface& s, const Medium& m1, const Medium& m2){
             RayVector result;
             result.rays = std::move(refract(rv.rays, s, m1, m2));
             return result;
         })
         .def("refractInPlace", (void (*)(Ray&, const Surface&, const double, const double)) &refractInPlace)
         .def("refractInPlace", [](RayVector& rv, const Surface& s, const double n1, const double n2){
             refractInPlace(rv.rays, s, n1, n2);
         })
         .def("refractInPlace", (void (*)(Ray&, const Surface&, const Medium&, const Medium&)) &refractInPlace)
         .def("refractInPlace", [](RayVector& rv, const Surface& s, const Medium& m1, const Medium& m2){
             refractInPlace(rv.rays, s, m1, m2);
         })

         .def("rayGrid",
             [](double zdist, double length, double xcos, double ycos, double zcos, int nside, double wavelength, double n)
                 {
                     RayVector rv;
                     rv.rays = std::move(rayGrid(zdist, length, xcos, ycos, zcos, nside, wavelength, n));
                     return rv;
                 },
             "Make a RayVector in a grid",
             "zdist"_a, "length"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nside"_a, "wavelength"_a, "n"_a
         )
         .def("rayGrid",
             [](double zdist, double length, double xcos, double ycos, double zcos, int nside, double wavelength, const Medium& m)
                 {
                     RayVector rv;
                     rv.rays = std::move(rayGrid(zdist, length, xcos, ycos, zcos, nside, wavelength, m));
                     return rv;
                 },
             "Make a RayVector in a grid",
             "zdist"_a, "length"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nside"_a, "wavelength"_a, "medium"_a
         )
         .def("circularGrid",
             [](double zdist, double outer, double inner, double xcos, double ycos, double zcos, int nradii, int naz, double wavelength, double n)
                 {
                     RayVector rv;
                     rv.rays = std::move(circularGrid(zdist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, n));
                     return rv;
                 },
             "Make a RayVector on a circle",
             "zdist"_a, "outer"_a, "inner"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nradii"_a, "naz"_a, "wavelength"_a, "n"_a
         )
         .def("circularGrid",
             [](double zdist, double outer, double inner, double xcos, double ycos, double zcos, int nradii, int naz, double wavelength, const Medium& m)
                 {
                     RayVector rv;
                     rv.rays = std::move(circularGrid(zdist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, m));
                     return rv;
                 },
             "Make a RayVector on a circle",
             "zdist"_a, "outer"_a, "inner"_a, "xcos"_a, "ycos"_a, "zcos"_a, "nradii"_a, "naz"_a, "wavelength"_a, "medium"_a
         )
         .def("trimVignetted", [](const RayVector& rv){
             RayVector result;
             result.rays = std::move(trimVignetted(rv.rays));
             return result;
         })
         .def("trimVignettedInPlace", [](RayVector& rv){
             trimVignettedInPlace(rv.rays);
         });

#if !((PYBIND11_VERSION_MAJOR >= 2) & (PYBIND11_VERSION_MINOR >= 2))
        return m.ptr();
#endif
    }
}
