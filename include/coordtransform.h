#ifndef batoid_coordtransform_h
#define batoid_coordtransform_h

#include "ray.h"
#include "rayVector.h"
#include "coordsys.h"
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace batoid {
    class CoordTransform {
    public:
        CoordTransform(const CoordSys& source, const CoordSys& destination);
        CoordTransform(const Vector3d& dr, const Matrix3d& rot);

        Vector3d applyForward(const Vector3d& r) const;
        Vector3d applyReverse(const Vector3d& r) const;

        Ray applyForward(const Ray& r) const;
        Ray applyReverse(const Ray& r) const;

        void applyForwardInPlace(Ray& r) const;
        void applyReverseInPlace(Ray& r) const;

        RayVector applyForward(const RayVector& rv) const;
        RayVector applyReverse(const RayVector& rv) const;
        void applyForwardInPlace(RayVector& rv) const;
        void applyReverseInPlace(RayVector& rv) const;

        const Matrix3d& getRot() const { return _rot; }
        const Vector3d& getDr() const { return _dr; }

        std::string repr() const {
            std::ostringstream oss;
            oss << "CoordTransform(" << _source << ", " << _destination << ")";
            return oss.str();
        }

        CoordSys getSource() const { return _source; }
        CoordSys getDestination() const { return _destination; }

    private:
        const Vector3d _dr;
        const Matrix3d _rot;
        const CoordSys _source;
        const CoordSys _destination;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);
    bool operator==(const CoordTransform& ct1, const CoordTransform& ct2);
    bool operator!=(const CoordTransform& ct1, const CoordTransform& ct2);
}

#endif
