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
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CoordSys")
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
                    "CoordSys",
                    py::tuple(py::cast(cs.m_origin)),
                    py::tuple(py::cast(cs.m_rot).attr("ravel")())
                ));
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

        auto resultX = py::array_t<double>(bufX.shape);
        auto resultY = py::array_t<double>(bufY.shape);
        auto resultZ = py::array_t<double>(bufZ.shape);
        auto bufXOut = resultX.request();
        auto bufYOut = resultY.request();
        auto bufZOut = resultZ.request();

        double *ptrXOut = (double *) bufXOut.ptr;
        double *ptrYOut = (double *) bufYOut.ptr;
        double *ptrZOut = (double *) bufZOut.ptr;

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(bufX.ndim, 0);
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            char *ptrX = (char *) bufX.ptr;
            char *ptrY = (char *) bufY.ptr;
            char *ptrZ = (char *) bufZ.ptr;
            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                ptrX += idxVec[idim]*bufX.strides[idim];
                ptrY += idxVec[idim]*bufY.strides[idim];
                ptrZ += idxVec[idim]*bufZ.strides[idim];
            }

            v = ct.applyForward(Vector3d(*(double *)ptrX, *(double *)ptrY, *(double *)ptrZ));

            ptrXOut[idx] = v[0];
            ptrYOut[idx] = v[1];
            ptrZOut[idx] = v[2];

            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == bufX.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }


    // Version of applyForward that accepts three congruent numpy arrays (x, y, z), and returns
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

        auto resultX = py::array_t<double>(bufX.shape);
        auto resultY = py::array_t<double>(bufY.shape);
        auto resultZ = py::array_t<double>(bufZ.shape);
        auto bufXOut = resultX.request();
        auto bufYOut = resultY.request();
        auto bufZOut = resultZ.request();

        double *ptrXOut = (double *) bufXOut.ptr;
        double *ptrYOut = (double *) bufYOut.ptr;
        double *ptrZOut = (double *) bufZOut.ptr;

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(bufX.ndim, 0);
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            char *ptrX = (char *) bufX.ptr;
            char *ptrY = (char *) bufY.ptr;
            char *ptrZ = (char *) bufZ.ptr;
            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                ptrX += idxVec[idim]*bufX.strides[idim];
                ptrY += idxVec[idim]*bufY.strides[idim];
                ptrZ += idxVec[idim]*bufZ.strides[idim];
            }

            v = ct.applyReverse(Vector3d(*(double *)ptrX, *(double *)ptrY, *(double *)ptrZ));

            ptrXOut[idx] = v[0];
            ptrYOut[idx] = v[1];
            ptrZOut[idx] = v[2];

            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == bufX.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }


    void pyExportCoordTransform(py::module& m) {
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CoordTransform")
            .def(py::init<const CoordSys&, const CoordSys&>())
            .def("applyForward", (Vector3d (CoordTransform::*)(const Vector3d&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Vector3d (CoordTransform::*)(const Vector3d&) const) &CoordTransform::applyReverse)
            .def("applyForward", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyForward(ct, xs, ys, zs);
            })
            .def("applyReverse", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyReverse(ct, xs, ys, zs);
            })
            .def("applyForward", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyReverse)
            .def("applyForward", (RayVector (CoordTransform::*)(const RayVector&) const) &CoordTransform::applyForward)
            .def("applyReverse", (RayVector (CoordTransform::*)(const RayVector&) const) &CoordTransform::applyReverse)
            .def("applyForwardInPlace", (void (CoordTransform::*)(Ray&) const) &CoordTransform::applyForwardInPlace)
            .def("applyForwardInPlace", (void (CoordTransform::*)(RayVector&) const) &CoordTransform::applyForwardInPlace)
            .def("applyReverseInPlace", (void (CoordTransform::*)(Ray&) const) &CoordTransform::applyReverseInPlace)
            .def("applyReverseInPlace", (void (CoordTransform::*)(RayVector&) const) &CoordTransform::applyReverseInPlace)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordTransform& ct) { return py::make_tuple(ct.getDr(), ct.getRot()); },
                [](py::tuple t) { return CoordTransform(t[0].cast<Vector3d>(), t[1].cast<Matrix3d>()); }
            ))
            .def("__hash__", [](CoordTransform& ct) {
                return py::hash(py::make_tuple(
                    "CoordTransform",
                    py::tuple(py::cast(ct.getDr())),
                    py::tuple(py::cast(ct.getRot()).attr("ravel")())
                ));
            })
            .def("__repr__", &CoordTransform::repr);
    }
}
