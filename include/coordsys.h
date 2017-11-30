#ifndef batoid_coordsys_h
#define batoid_coordsys_h

#include "vec3.h"
#include "ray.h"
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace batoid {
    struct CoordSys {
        // explicitly construct the global coordinate system
        CoordSys();

        // _origin indicates the origin in global coordinates of the new coordinate system.
        // _rotation indicates the rotation applied to the global unit vectors to produce
        // the unit vectors of the new coordinate system.
        CoordSys(Vec3 _origin, Rot3 _rotation);

        // Could add an Euler angle ctor too...

        // Create a new CoordSys with parallel axes, but with origin shifted by dr,
        // where dr is in global coordinates.
        CoordSys shiftGlobal(const Vec3& dr) const;

        // Same, but here dr is in the local coordinate system.
        CoordSys shiftLocal(const Vec3& dr) const;

        // Rotate wrt the global axes, where the center of rotation is the global origin.
        CoordSys rotateGlobal(const Rot3& rotation) const;

        // Rotate wrt the global axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateGlobal(const Rot3& rotation, const Vec3& rotOrigin, const CoordSys& coordSys) const;

        // Rotate wrt the local axes, around the local origin
        CoordSys rotateLocal(const Rot3& rotation) const;

        // Rotate wrt the local axes, around the given center of rotation, which is expressed
        // in the given coordinate system.
        CoordSys rotateLocal(const Rot3& rotation, const Vec3& rotOrigin, const CoordSys& coordSys) const;

        // Get local unit vectors in global coordinates.
        Vec3 getX() const;
        Vec3 getY() const;
        Vec3 getZ() const;

        std::string repr() const {
            std::ostringstream oss(" ");
            oss << "CoordSys(" << origin << ", " << rotation << ')';
            return oss.str();
        }

        const Vec3 origin;
        // Could potentially use Euler angles instead of rotation matrix here to be
        // more compact?
        const Rot3 rotation;
    };

    std::ostream& operator<<(std::ostream &os, const CoordSys& cs);

    class BaseCoordTransform {
    public:
        virtual Vec3 applyForward(const Vec3& r) const = 0;
        virtual Vec3 applyReverse(const Vec3& r) const = 0;

        virtual Ray applyForward(const Ray& r) const = 0;
        virtual Ray applyReverse(const Ray& r) const = 0;

        virtual void applyForwardInPlace(Ray& r) const = 0;
        virtual void applyReverseInPlace(Ray& r) const = 0;

        std::vector<Ray> applyForward(const std::vector<Ray>& r) const;
        std::vector<Ray> applyReverse(const std::vector<Ray>& r) const;
        void applyForwardInPlace(std::vector<Ray>& rays) const;
        void applyReverseInPlace(std::vector<Ray>& rays) const;

        virtual Rot3 getRot() const = 0;
        virtual Vec3 getDr() const = 0;
    };

    class IdentityTransform : public BaseCoordTransform {
    public:
        IdentityTransform() = default;
        virtual Vec3 applyForward(const Vec3& r) const override { return r; }
        virtual Vec3 applyReverse(const Vec3& r) const override { return r; }

        virtual Ray applyForward(const Ray& r) const override { return r; }
        virtual Ray applyReverse(const Ray& r) const override { return r; }

        virtual void applyForwardInPlace(Ray& r) const override {}  // noop
        virtual void applyReverseInPlace(Ray& r) const override {}

        virtual Rot3 getRot() const override { return Rot3(); }
        virtual Vec3 getDr() const override { return Vec3(); }
    };

    class ShiftTransform : public BaseCoordTransform {
    public:
        ShiftTransform(Vec3 dr) : _dr(dr) {}
        virtual Vec3 applyForward(const Vec3& r) const override { return r-_dr; }
        virtual Vec3 applyReverse(const Vec3& r) const override { return r+_dr; }

        virtual Ray applyForward(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(r.p0-_dr, r.v, r.t0, r.wavelength, r.isVignetted);
        }
        virtual Ray applyReverse(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(r.p0+_dr, r.v, r.t0, r.wavelength, r.isVignetted);
        }

        virtual void applyForwardInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 -= _dr;
        }
        virtual void applyReverseInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 += _dr;
        }

        virtual Rot3 getRot() const override { return Rot3(); }
        virtual Vec3 getDr() const override { return _dr; }
    private:
        const Vec3 _dr;
    };

    class RotTransform : public BaseCoordTransform {
    public:
        RotTransform(Rot3 rot) : _rot(rot) {}
        virtual Vec3 applyForward(const Vec3& r) const override { return UnRotVec(_rot, r); }
        virtual Vec3 applyReverse(const Vec3& r) const override { return RotVec(_rot, r); }

        virtual Ray applyForward(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(UnRotVec(_rot, r.p0), UnRotVec(_rot, r.v),
                       r.t0, r.wavelength, r.isVignetted);
        }
        virtual Ray applyReverse(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(RotVec(_rot, r.p0), RotVec(_rot, r.v),
                       r.t0, r.wavelength, r.isVignetted);
        }

        virtual void applyForwardInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 = UnRotVec(_rot, r.p0);
            r.v = UnRotVec(_rot, r.v);
        }
        virtual void applyReverseInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 = RotVec(_rot, r.p0);
            r.v = RotVec(_rot, r.v);
        }

        virtual Rot3 getRot() const override { return _rot; }
        virtual Vec3 getDr() const override { return Vec3(); }
    private:
        const Rot3 _rot;
    };

    class CoordTransform : public BaseCoordTransform {
    public:
        CoordTransform(Vec3 dr, Rot3 rot) : _dr(dr), _rot(rot) {}

        // We actively shift and rotate the coordinate system axes,
        // This looks like y = R x + dr
        // For a passive transformation of a fixed vector from one coord sys to another
        // though, we want the opposite transformation: y = R^-1 (x - dr)
        virtual Vec3 applyForward(const Vec3& r) const override { return UnRotVec(_rot, r-_dr); }
        virtual Vec3 applyReverse(const Vec3& r) const override { return RotVec(_rot, r)+_dr; }

        virtual Ray applyForward(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(UnRotVec(_rot, r.p0-_dr), UnRotVec(_rot, r.v),
                    r.t0, r.wavelength, r.isVignetted);
        }
        virtual Ray applyReverse(const Ray& r) const override {
            if (r.failed) return r;
            return Ray(RotVec(_rot, r.p0) + _dr, RotVec(_rot, r.v),
                r.t0, r.wavelength, r.isVignetted);
        }

        virtual void applyForwardInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 = UnRotVec(_rot, r.p0-_dr);
            r.v = UnRotVec(_rot, r.v);
        }
        virtual void applyReverseInPlace(Ray& r) const override {
            if (r.failed) return;
            r.p0 = RotVec(_rot, r.p0)+_dr;
            r.v = RotVec(_rot, r.v);
        }

        virtual Rot3 getRot() const override { return _rot; }
        virtual Vec3 getDr() const override { return _dr; }
    private:
        const Vec3 _dr;
        const Rot3 _rot;
    };

    std::unique_ptr<BaseCoordTransform> getTransform(const CoordSys& source, const CoordSys& destination);

}

#endif
