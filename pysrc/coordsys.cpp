#include "coordsys.h"
#include <pybind11/pybind11.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Vec3>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportCoordSys(py::module& m) {
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CoordSys")
            .def(py::init<>())
            .def(py::init<Vec3,Rot3>())
            .def_readonly("origin", &CoordSys::origin, "Global origin")
            .def_readonly("rotation", &CoordSys::rotation, "Unit vector rotation matrix")
            .def_property_readonly("x", &CoordSys::getX)
            .def_property_readonly("y", &CoordSys::getY)
            .def_property_readonly("z", &CoordSys::getZ)
            .def("__repr__", &CoordSys::repr)
            .def("shiftGlobal", &CoordSys::shiftGlobal)
            .def("shiftLocal", &CoordSys::shiftLocal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Rot3&) const) &CoordSys::rotateGlobal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, std::shared_ptr<CoordSys>) const) &CoordSys::rotateGlobal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&) const) &CoordSys::rotateLocal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, std::shared_ptr<CoordSys>) const) &CoordSys::rotateLocal);

        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CoordTransform")
            .def(py::init<std::shared_ptr<CoordSys>,std::shared_ptr<CoordSys>>())
            .def("applyForward", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyReverse)
            .def("applyForward", (std::vector<Vec3> (CoordTransform::*)(const std::vector<Vec3>&) const) &CoordTransform::applyForward)
            .def("applyReverse", (std::vector<Vec3> (CoordTransform::*)(const std::vector<Vec3>&) const) &CoordTransform::applyReverse)
            .def("applyForward", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyReverse)
            .def("applyForward", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyForward)
            .def("applyReverse", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyReverse)
            .def_readonly("source", &CoordTransform::source)
            .def_readonly("destination", &CoordTransform::destination);
    }
}
