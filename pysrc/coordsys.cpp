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
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, const CoordSys&) const) &CoordSys::rotateGlobal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&) const) &CoordSys::rotateLocal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, const CoordSys&) const) &CoordSys::rotateLocal);
    }


    // Version of applyForward that accepts three congruent numpy arrays (x, y, z), and returns
    // three transformed numpy arrays with the new coordinates.
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    numpyApplyForward(const BaseCoordTransform& ct,
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
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            v = ct.applyForward(Vec3(ptrX[idx], ptrY[idx], ptrZ[idx]));
            ptrXOut[idx] = v.x;
            ptrYOut[idx] = v.y;
            ptrZOut[idx] = v.z;
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }

    // Version of applyReverse that accepts three congruent numpy arrays (x, y, z), and returns
    // three transformed numpy arrays with the new coordinates.
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    numpyApplyReverse(const BaseCoordTransform& ct,
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
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            v = ct.applyReverse(Vec3(ptrX[idx], ptrY[idx], ptrZ[idx]));
            ptrXOut[idx] = v.x;
            ptrYOut[idx] = v.y;
            ptrZOut[idx] = v.z;
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }

    void pyExportCoordTransform(py::module& m) {
        py::class_<BaseCoordTransform, std::shared_ptr<BaseCoordTransform>>(m, "CoordTransform")
            .def(py::init(&getTransform))
            .def("applyForward", (Vec3 (BaseCoordTransform::*)(const Vec3&) const) &BaseCoordTransform::applyForward)
            .def("applyReverse", (Vec3 (BaseCoordTransform::*)(const Vec3&) const) &BaseCoordTransform::applyReverse)
            .def("applyForward", [](const BaseCoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyForward(ct, xs, ys, zs);
            })
            .def("applyReverse", [](const BaseCoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyReverse(ct, xs, ys, zs);
            })
            .def("applyForward", (Ray (BaseCoordTransform::*)(const Ray&) const) &BaseCoordTransform::applyForward)
            .def("applyReverse", (Ray (BaseCoordTransform::*)(const Ray&) const) &BaseCoordTransform::applyReverse)
            .def("applyForward", (std::vector<Ray> (BaseCoordTransform::*)(const std::vector<Ray>&) const) &BaseCoordTransform::applyForward)
            .def("applyReverse", (std::vector<Ray> (BaseCoordTransform::*)(const std::vector<Ray>&) const) &BaseCoordTransform::applyReverse)
            .def("applyForwardInPlace", (void (BaseCoordTransform::*)(Ray&) const) &BaseCoordTransform::applyForwardInPlace)
            .def("applyForwardInPlace", (void (BaseCoordTransform::*)(std::vector<Ray>&) const) &BaseCoordTransform::applyForwardInPlace)
            ;
    }
}
