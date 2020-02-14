#include "coordsys.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <tuple>
#include <iostream>

namespace py = pybind11;

namespace batoid {
    void pyExportCoordSys(py::module& m) {
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CPPCoordSys")
            .def(py::init<>())
            .def(py::init<Vector3d>())
            .def(py::init<Matrix3d>())
            .def(py::init<Vector3d,Matrix3d>())
            .def_readonly("origin", &CoordSys::m_origin, "Global origin")
            .def_readonly("rot", &CoordSys::m_rot, "Unit vector rotation matrix")
            .def_property_readonly("xhat", &CoordSys::getXHat)
            .def_property_readonly("yhat", &CoordSys::getYHat)
            .def_property_readonly("zhat", &CoordSys::getZHat)
            .def("__repr__", &CoordSys::repr)
            .def("shiftGlobal", &CoordSys::shiftGlobal)
            .def("shiftLocal", &CoordSys::shiftLocal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Matrix3d&) const) &CoordSys::rotateGlobal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Matrix3d&, const Vector3d&, const CoordSys&) const) &CoordSys::rotateGlobal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Matrix3d&) const) &CoordSys::rotateLocal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Matrix3d&, const Vector3d&, const CoordSys&) const) &CoordSys::rotateLocal)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordSys& cs) { return py::make_tuple(cs.m_origin, cs.m_rot); },
                [](py::tuple t) {
                    return CoordSys(
                        t[0].cast<Vector3d>(), t[1].cast<Matrix3d>()
                    );
                }
            ))
            .def("__hash__", [](const CoordSys& cs) {
                return py::hash(py::make_tuple(
                    "CPPCoordSys",
                    py::tuple(py::cast(cs.m_origin)),
                    py::tuple(py::cast(cs.m_rot).attr("ravel")())
                ));
            });
    }
}
