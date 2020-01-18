#include "medium2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportMedium2(py::module& m) {
        py::class_<Medium2>(m, "CPPMedium2");

        py::class_<SellmeierMedium2, Medium2>(m, "CPPSellmeierMedium2")
            .def(py::init<double,double,double,double,double,double>())
            .def("getN", &SellmeierMedium2::getN)
            .def("getNMany",
                [](const SellmeierMedium2& m, size_t n, size_t wavelength_ptr, size_t out_ptr)
                {
                    m.getNMany(
                        n,
                        reinterpret_cast<double*>(wavelength_ptr),
                        reinterpret_cast<double*>(out_ptr)
                    );
                }
            );
    }
}
