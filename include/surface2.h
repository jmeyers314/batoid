#ifndef batoid_surface2_h
#define batoid_surface2_h

#include <vector>
#include <memory>
#include <utility>
#include "rayVector2.h"
#include "medium2.h"
//#include "coating.h"

#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    class Surface2 {
    public:
        virtual ~Surface2() {}

        virtual void intersectInPlace(RayVector2&) const = 0;
        virtual void reflectInPlace(RayVector2&) const = 0;
        virtual void refractInPlace(RayVector2&, const Medium2&, const Medium2&) const = 0;
    };
}
#endif
