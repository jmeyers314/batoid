#include "coordSys.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    using vec3 = CoordSys::vec3;
    using mat3 = CoordSys::mat3;

    void pyExportCoordSys(py::module& m) {
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CPPCoordSys")
            .def(py::init<>())
            .def(py::init<vec3>())
            .def(py::init<mat3>())
            .def(py::init<vec3, mat3>())
            .def_readonly("origin", &CoordSys::m_origin, "Global origin")
            .def_readonly("rot", &CoordSys::m_rot, "Unit vector rotation matrix")
            .def_property_readonly("xhat", &CoordSys::getXHat)
            .def_property_readonly("yhat", &CoordSys::getYHat)
            .def_property_readonly("zhat", &CoordSys::getZHat)
            .def("shiftGlobal", &CoordSys::shiftGlobal)
            .def("shiftLocal", &CoordSys::shiftLocal)
            .def("rotateGlobal", py::overload_cast<const mat3&>(&CoordSys::rotateGlobal, py::const_))
            .def("rotateGlobal", py::overload_cast<const mat3&, const vec3&, const CoordSys&>(&CoordSys::rotateGlobal, py::const_))
            .def("rotateLocal", py::overload_cast<const mat3&>(&CoordSys::rotateLocal, py::const_))
            .def("rotateLocal", py::overload_cast<const mat3&, const vec3&, const CoordSys&>(&CoordSys::rotateLocal, py::const_));
    }
}
