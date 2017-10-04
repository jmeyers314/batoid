#include "coordsys.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>

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
    }

    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    numpyApplyForward(const CoordTransform& ct,
                      py::array_t<double> xs,
                      py::array_t<double> ys,
                      py::array_t<double> zs) {
        auto bufX = xs.request();
        auto bufY = ys.request();
        auto bufZ = zs.request();
        if (bufX.ndim != bufY.ndim || bufX.ndim != bufZ.ndim)
            throw std::runtime_error("Dimensions must match");
        if (bufX.size != bufY.size || bufX.size != bufZ.size)
            throw std::runtime_error("Sizes much match");
        auto resultX = py::array_t<double>(bufX.shape, bufX.strides);
        auto resultY = py::array_t<double>(bufY.shape, bufY.strides);
        auto resultZ = py::array_t<double>(bufZ.shape, bufZ.strides);
        auto bufXOut = resultX.request();
        auto bufYOut = resultY.request();
        auto bufZOut = resultZ.request();

        double *ptrX = (double *) bufX.ptr;
        double *ptrY = (double *) bufY.ptr;
        double *ptrZ = (double *) bufZ.ptr;
        double *ptrXOut = (double *)  bufXOut.ptr;
        double *ptrYOut = (double *)  bufYOut.ptr;
        double *ptrZOut = (double *)  bufZOut.ptr;

        auto v = Vec3();
        for (size_t idx = 0; idx < bufX.size; idx++) {
            v = ct.applyForward(Vec3(ptrX[idx], ptrY[idx], ptrZ[idx]));
            ptrXOut[idx] = v.x;
            ptrYOut[idx] = v.y;
            ptrZOut[idx] = v.z;
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }

    void pyExportCoordTransform(py::module& m) {
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CoordTransform")
            .def(py::init<std::shared_ptr<CoordSys>,std::shared_ptr<CoordSys>>())
            .def("applyForward", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyReverse)
            .def("applyForward", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyForward(ct, xs, ys, zs);
            })
            .def("applyForward", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyReverse)
            .def("applyForward", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyForward)
            .def("applyReverse", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyReverse)
            .def_readonly("source", &CoordTransform::source)
            .def_readonly("destination", &CoordTransform::destination);
    }
}
