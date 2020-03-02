#include "surface2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface2(py::module& m) {
        py::class_<Surface2, std::shared_ptr<Surface2>>(m, "CPPSurface2")
            .def("intersectInPlace", &Surface2::intersectInPlace, py::arg(), py::arg()=nullptr)
            .def("reflectInPlace", &Surface2::reflectInPlace, py::arg(), py::arg()=nullptr)
            .def("refractInPlace", &Surface2::refractInPlace, py::arg(), py::arg(), py::arg(), py::arg()=nullptr)
            .def("sag",
                [](
                    const Surface2& s,
                    size_t xptr,
                    size_t yptr,
                    size_t size,
                    size_t outptr
                ){
                    s.sag(
                        reinterpret_cast<double*>(xptr),
                        reinterpret_cast<double*>(yptr),
                        size,
                        reinterpret_cast<double*>(outptr)
                    );
                }
            )
            .def("normal",
                [](
                    const Surface2& s,
                    size_t xptr,
                    size_t yptr,
                    size_t size,
                    size_t outptr
                ){
                    s.normal(
                        reinterpret_cast<double*>(xptr),
                        reinterpret_cast<double*>(yptr),
                        size,
                        reinterpret_cast<double*>(outptr)
                    );
                }
            );
    }
}
