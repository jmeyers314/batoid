#include "coordsys.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <tuple>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Vec3>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportCoordSys(py::module& m) {
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CoordSys")
            .def(py::init<>())
            .def(py::init<Vec3>())
            .def(py::init<Rot3>())
            .def(py::init<Vec3,Rot3>())
            .def_readonly("origin", &CoordSys::origin, "Global origin")
            .def_readonly("rotation", &CoordSys::rotation, "Unit vector rotation matrix")
            .def_property_readonly("xhat", &CoordSys::getXHat)
            .def_property_readonly("yhat", &CoordSys::getYHat)
            .def_property_readonly("zhat", &CoordSys::getZHat)
            .def("__repr__", &CoordSys::repr)
            .def("shiftGlobal", &CoordSys::shiftGlobal)
            .def("shiftLocal", &CoordSys::shiftLocal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Rot3&) const) &CoordSys::rotateGlobal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, const CoordSys&) const) &CoordSys::rotateGlobal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&) const) &CoordSys::rotateLocal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Rot3&, const Vec3&, const CoordSys&) const) &CoordSys::rotateLocal)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordSys& cs) { return py::make_tuple(cs.origin, cs.rotation); },
                [](py::tuple t) {
                    return CoordSys(
                        t[0].cast<Vec3>(), t[1].cast<Rot3>()
                    );
                }
            ))
            .def("__hash__", [](const CoordSys& cs) {
                return py::hash(py::make_tuple("CoordSys", cs.origin, cs.rotation));
            });
    }


    // Version of applyForward that accepts three congruent numpy arrays (x, y, z), and returns
    // three transformed numpy arrays with the new coordinates.
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
    numpyApplyReverse(const CoordTransform& ct,
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
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CoordTransform")
            .def(py::init<const CoordSys&, const CoordSys&>())
            .def("applyForward", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Vec3 (CoordTransform::*)(const Vec3&) const) &CoordTransform::applyReverse)
            .def("applyForward", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyForward(ct, xs, ys, zs);
            })
            .def("applyReverse", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyReverse(ct, xs, ys, zs);
            })
            .def("applyForward", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyReverse)
            .def("applyForward", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyForward)
            .def("applyReverse", (std::vector<Ray> (CoordTransform::*)(const std::vector<Ray>&) const) &CoordTransform::applyReverse)
            .def("applyForwardInPlace", (void (CoordTransform::*)(Ray&) const) &CoordTransform::applyForwardInPlace)
            .def("applyForwardInPlace", (void (CoordTransform::*)(std::vector<Ray>&) const) &CoordTransform::applyForwardInPlace)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordTransform& ct) { return py::make_tuple(ct.getDr(), ct.getRot()); },
                [](py::tuple t) { return CoordTransform(t[0].cast<Vec3>(), t[1].cast<Rot3>()); }
            ))
            .def("__hash__", [](CoordTransform& ct) {
                return py::hash(py::make_tuple("CoordTransform", ct.getDr(), ct.getRot()));
            })
            .def("__repr__", &CoordTransform::repr);
    }
}
