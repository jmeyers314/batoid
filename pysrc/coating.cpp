#include "coating.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;


namespace batoid {
    void pyExportCoating(py::module& m) {
        py::class_<CoatingHandle, std::shared_ptr<CoatingHandle>>(m, "CPPCoating")
            .def("getCoefs", [](const CoatingHandle& handle, double wavelength, double cosIncidenceAngle){
                double reflect, transmit;
                handle.getHostPtr()->getCoefs(wavelength, cosIncidenceAngle, reflect, transmit);
                return py::make_tuple(reflect, transmit);
            })
            .def("getReflect", [](const CoatingHandle& handle, double wavelength, double cosIncidenceAngle){
                return handle.getHostPtr()->getReflect(wavelength, cosIncidenceAngle);
            })
            .def("getTransmit", [](const CoatingHandle& handle, double wavelength, double cosIncidenceAngle){
                return handle.getHostPtr()->getTransmit(wavelength, cosIncidenceAngle);
            });

        py::class_<SimpleCoatingHandle, std::shared_ptr<SimpleCoatingHandle>, CoatingHandle>(m, "CPPSimpleCoating")
            .def(py::init<double,double>(), "reflectivity"_a, "transmissivity"_a);
    }
}
