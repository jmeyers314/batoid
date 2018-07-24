#include "zernike.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportZernike(py::module& m) {
        m.def("nCr", &zernike::nCr);
        m.def("binomial", &zernike::binomial);
        m.def("horner2d", &zernike::horner2d);
        m.def("noll_to_zern", &zernike::noll_to_zern);

        py::class_<Zernike, std::shared_ptr<Zernike>, Surface>(m, "Zernike")
            .def(py::init<std::vector<double>,double,double>(), "init", "coefs"_a, "R_outer"_a=1.0, "R_inner"_a=0.0)
            .def_property_readonly("coefs", &Zernike::getCoefs)
            .def_property_readonly("R_outer", &Zernike::getROuter)
            .def_property_readonly("R_inner", &Zernike::getRInner)
            .def_property_readonly("gradX", &Zernike::getGradX)
            .def_property_readonly("gradY", &Zernike::getGradY)
            .def(py::pickle(
                [](const Zernike& z){ return py::make_tuple(z.getCoefs(), z.getROuter(), z.getRInner()); },
                [](py::tuple t) {
                    return Zernike(
                        t[0].cast<std::vector<double>>(),
                        t[1].cast<double>(),
                        t[2].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const Zernike& z) {
                return py::hash(py::make_tuple(
                    "Zernike",
                    py::tuple(py::cast(z.getCoefs())),
                    z.getROuter(),
                    z.getRInner()
                ));
            });

    }
}
