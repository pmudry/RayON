/**
 * @file sdf.h
 * @brief Signed Distance Field (SDF) functions and ray marching utilities
 * 
 * This file provides:
 * - Distance functions for various 3D primitives (sphere, box, torus, etc.)
 * - SDF operations (union, subtraction, intersection, smooth blending)
 * - Ray marching algorithm (sphere tracing)
 * - Normal estimation using gradient method
 * 
 * SDFs represent surfaces implicitly as the zero level-set of a distance function.
 * Ray marching traces rays by iteratively stepping along the ray direction,
 * using the distance field to safely advance without missing intersections.
 */

#pragma once

#include "vec3.hpp"
#include "ray.hpp"
#include "interval.hpp"
#include <cmath>
#include <algorithm>

namespace SDF {

//==============================================================================
// ROTATION UTILITIES
//==============================================================================

/**
 * @brief Rotate point around X axis
 */
inline Vec3 rotateX(const Vec3& p, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Vec3(p.x(), c * p.y() - s * p.z(), s * p.y() + c * p.z());
}

/**
 * @brief Rotate point around Y axis
 */
inline Vec3 rotateY(const Vec3& p, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Vec3(c * p.x() + s * p.z(), p.y(), -s * p.x() + c * p.z());
}

/**
 * @brief Rotate point around Z axis
 */
inline Vec3 rotateZ(const Vec3& p, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Vec3(c * p.x() - s * p.y(), s * p.x() + c * p.y(), p.z());
}

/**
 * @brief Apply Euler rotation (X, Y, Z order)
 * @param p Point to rotate
 * @param rotation Rotation angles in radians (X, Y, Z)
 */
inline Vec3 applyRotation(const Vec3& p, const Vec3& rotation) {
    Vec3 result = p;
    if (rotation.x() != 0.0) result = rotateX(result, rotation.x());
    if (rotation.y() != 0.0) result = rotateY(result, rotation.y());
    if (rotation.z() != 0.0) result = rotateZ(result, rotation.z());
    return result;
}

/**
 * @brief Apply inverse Euler rotation (reverse order: Z, Y, X with negated angles)
 * Used to transform points into the SDF's local space
 */
inline Vec3 applyInverseRotation(const Vec3& p, const Vec3& rotation) {
    Vec3 result = p;
    if (rotation.z() != 0.0) result = rotateZ(result, -rotation.z());
    if (rotation.y() != 0.0) result = rotateY(result, -rotation.y());
    if (rotation.x() != 0.0) result = rotateX(result, -rotation.x());
    return result;
}

//==============================================================================
// SDF PRIMITIVE DISTANCE FUNCTIONS
//==============================================================================

/**
 * @brief Signed distance to a sphere
 * @param p Point in space
 * @param center Center of sphere
 * @param radius Radius of sphere
 * @return Signed distance (negative inside, positive outside, zero on surface)
 */
inline double sdSphere(const Vec3& p, const Vec3& center, double radius) {
    return (p - center).length() - radius;
}

/**
 * @brief Signed distance to an axis-aligned box
 * @param p Point in space
 * @param center Center of box
 * @param size Half-extents of box (width/2, height/2, depth/2)
 * @return Signed distance
 */
inline double sdBox(const Vec3& p, const Vec3& center, const Vec3& size) {
    Vec3 offset = p - center;
    Vec3 q = Vec3(std::abs(offset.x()), std::abs(offset.y()), std::abs(offset.z())) - size;
    
    // Distance outside + distance inside
    double outside = Vec3(std::max(q.x(), 0.0), std::max(q.y(), 0.0), std::max(q.z(), 0.0)).length();
    double inside = std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0);
    
    return outside + inside;
}

/**
 * @brief Signed distance to a torus
 * @param p Point in space
 * @param center Center of torus
 * @param major_radius Distance from center to tube center (big radius)
 * @param minor_radius Radius of the tube itself (small radius)
 * @return Signed distance
 */
inline double sdTorus(const Vec3& p, const Vec3& center, double major_radius, double minor_radius) {
    Vec3 offset = p - center;
    
    // Distance from point to center in XZ plane
    double q_x = std::sqrt(offset.x() * offset.x() + offset.z() * offset.z()) - major_radius;
    
    // Distance to torus tube
    return std::sqrt(q_x * q_x + offset.y() * offset.y()) - minor_radius;
}

/**
 * @brief Signed distance to a capsule (cylinder with hemispherical caps)
 * @param p Point in space
 * @param a Start point of capsule axis
 * @param b End point of capsule axis
 * @param radius Radius of capsule
 * @return Signed distance
 */
inline double sdCapsule(const Vec3& p, const Vec3& a, const Vec3& b, double radius) {
    Vec3 pa = p - a;
    Vec3 ba = b - a;
    
    // Project point onto line segment
    double h = std::clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    
    // Distance to closest point on segment
    return (pa - ba * h).length() - radius;
}

/**
 * @brief Signed distance to a cylinder
 * @param p Point in space
 * @param center Center of cylinder
 * @param height Height of cylinder
 * @param radius Radius of cylinder
 * @return Signed distance
 */
inline double sdCylinder(const Vec3& p, const Vec3& center, double height, double radius) {
    Vec3 offset = p - center;
    
    // Distance to cylindrical surface
    double d_xz = std::sqrt(offset.x() * offset.x() + offset.z() * offset.z()) - radius;
    double d_y = std::abs(offset.y()) - height * 0.5;
    
    // Combine distances
    double outside = Vec3(std::max(d_xz, 0.0), std::max(d_y, 0.0), 0.0).length();
    double inside = std::min(std::max(d_xz, d_y), 0.0);
    
    return outside + inside;
}

/**
 * @brief Signed distance to a plane
 * @param p Point in space
 * @param normal Normal vector of plane (should be normalized)
 * @param distance Distance from origin along normal
 * @return Signed distance
 */
inline double sdPlane(const Vec3& p, const Vec3& normal, double distance) {
    return dot(p, normal) + distance;
}

/**
 * @brief Signed distance to a Mandelbulb fractal
 * @param p Point in space
 * @param center Center of mandelbulb
 * @param power Fractal power (typically 8)
 * @param iterations Maximum iterations
 * @return Estimated distance
 */
inline double sdMandelbulb(const Vec3& p, const Vec3& center, double power = 8.0, int iterations = 15) {
    Vec3 z = p - center;
    double dr = 1.0;
    double r = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        r = z.length();
        
        if (r > 2.0) break;
        
        // Convert to polar coordinates
        double theta = std::acos(z.z() / r);
        double phi = std::atan2(z.y(), z.x());
        dr = std::pow(r, power - 1.0) * power * dr + 1.0;
        
        // Scale and rotate the point
        double zr = std::pow(r, power);
        theta = theta * power;
        phi = phi * power;
        
        // Convert back to cartesian coordinates
        z = zr * Vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(phi) * std::sin(theta),
            std::cos(theta)
        );
        z = z + (p - center);
    }
    
    return 0.5 * std::log(r) * r / dr;
}

/**
 * @brief Signed distance to a Death Star shape (sphere with spherical cutout)
 * @param p Point in space
 * @param center Center of the death star
 * @param ra Radius of main sphere
 * @param rb Radius of spherical cutout
 * @param d Distance from center to cutout sphere center
 * @return Signed distance
 * 
 * Based on Íñigo Quilez's distance functions
 * https://iquilezles.org/articles/distfunctions/
 */
inline double sdDeathStar(const Vec3& p, const Vec3& center, double ra, double rb, double d) {
    Vec3 offset = p - center;
    
    // Calculate intersection parameters
    double a = (ra*ra - rb*rb + d*d) / (2.0*d);
    double b = std::sqrt(std::max(ra*ra - a*a, 0.0));
    
    // Work in 2D (project to XY plane, length of YZ)
    Vec3 p2 = Vec3(offset.x(), std::sqrt(offset.y()*offset.y() + offset.z()*offset.z()), 0.0);
    
    if (p2.x() * b - p2.y() * a > d * std::max(b - p2.y(), 0.0)) {
        return (p2 - Vec3(a, b, 0.0)).length();
    } else {
        return std::max(
            p2.length() - ra,
            -(p2 - Vec3(d, 0.0, 0.0)).length() + rb
        );
    }
}

/**
 * @brief Signed distance to a cut hollow sphere
 * @param p Point in space
 * @param center Center of the sphere
 * @param r Radius of sphere
 * @param h Height of the cut (from center)
 * @param t Thickness of the shell
 * @return Signed distance
 * 
 * Based on Íñigo Quilez's distance functions
 * https://iquilezles.org/articles/distfunctions/
 */
inline double sdCutHollowSphere(const Vec3& p, const Vec3& center, double r, double h, double t) {
    Vec3 offset = p - center;
    
    // Calculate width at cut height
    double w = std::sqrt(r*r - h*h);
    
    // Work in 2D (length in XZ plane, Y coordinate)
    Vec3 q = Vec3(std::sqrt(offset.x()*offset.x() + offset.z()*offset.z()), offset.y(), 0.0);
    
    double dist;
    if (h * q.x() < w * q.y()) {
        dist = (q - Vec3(w, h, 0.0)).length();
    } else {
        dist = std::abs(q.length() - r);
    }
    
    return dist - t;
}

/**
 * @brief Signed distance to an octahedron
 * @param p Point in space
 * @param center Center of the octahedron
 * @param s Size parameter (distance from center to vertex)
 * @return Signed distance
 * 
 * Based on Íñigo Quilez's distance functions
 * https://iquilezles.org/articles/distfunctions/
 */
inline double sdOctahedron(const Vec3& p, const Vec3& center, double s) {
    Vec3 offset = p - center;
    Vec3 pp = Vec3(std::abs(offset.x()), std::abs(offset.y()), std::abs(offset.z()));
    
    double m = pp.x() + pp.y() + pp.z() - s;
    
    Vec3 q;
    if (3.0 * pp.x() < m) {
        q = pp;
    } else if (3.0 * pp.y() < m) {
        q = Vec3(pp.y(), pp.z(), pp.x());
    } else if (3.0 * pp.z() < m) {
        q = Vec3(pp.z(), pp.x(), pp.y());
    } else {
        return m * 0.57735027;
    }
    
    double k = std::clamp(0.5 * (q.z() - q.y() + s), 0.0, s);
    return Vec3(q.x(), q.y() - s + k, q.z() - k).length();
}

/**
 * @brief Signed distance to a pyramid (square base)
 * @param p Point in space
 * @param center Center of the pyramid base
 * @param h Height of the pyramid
 * @return Signed distance
 * 
 * Based on Íñigo Quilez's distance functions
 * https://iquilezles.org/articles/distfunctions/
 * The pyramid has a square base of side length 1 centered at the origin
 */
inline double sdPyramid(const Vec3& p, const Vec3& center, double h) {
    Vec3 offset = p - center;
    
    double m2 = h*h + 0.25;
    
    // Mirror to first octant
    Vec3 pp = Vec3(std::abs(offset.x()), offset.y(), std::abs(offset.z()));
    
    // Swap x and z if needed
    if (pp.z() > pp.x()) {
        pp = Vec3(pp.z(), pp.y(), pp.x());
    }
    pp = pp - Vec3(0.5, 0.0, 0.5);
    
    // Project to 2D
    Vec3 q = Vec3(pp.z(), h*pp.y() - 0.5*pp.x(), h*pp.x() + 0.5*pp.y());
    
    double s = std::max(-q.x(), 0.0);
    double t = std::clamp((q.y() - 0.5*pp.z()) / (m2 + 0.25), 0.0, 1.0);
    
    double a = m2 * (q.x() + s) * (q.x() + s) + q.y() * q.y();
    double b = m2 * (q.x() + 0.5*t) * (q.x() + 0.5*t) + (q.y() - m2*t) * (q.y() - m2*t);
    
    double d2 = (std::min(q.y(), -q.x()*m2 - q.y()*0.5) > 0.0) ? 0.0 : std::min(a, b);
    
    double result = std::sqrt((d2 + q.z()*q.z()) / m2);
    return result * ((q.z() > 0.0 || pp.y() < 0.0) ? 1.0 : -1.0);
}

//==============================================================================
// SDF OPERATIONS
//==============================================================================

/**
 * @brief Union of two SDFs (returns closest surface)
 */
inline double opUnion(double d1, double d2) {
    return std::min(d1, d2);
}

/**
 * @brief Subtraction (d1 - d2)
 */
inline double opSubtraction(double d1, double d2) {
    return std::max(d1, -d2);
}

/**
 * @brief Intersection of two SDFs
 */
inline double opIntersection(double d1, double d2) {
    return std::max(d1, d2);
}

/**
 * @brief Smooth union using polynomial blending
 * @param d1 First distance
 * @param d2 Second distance
 * @param k Smoothing factor (larger = smoother)
 */
inline double opSmoothUnion(double d1, double d2, double k) {
    double h = std::clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h);
}

/**
 * @brief Smooth subtraction
 */
inline double opSmoothSubtraction(double d1, double d2, double k) {
    double h = std::clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return d2 * (1.0 - h) - d1 * h + k * h * (1.0 - h);
}

/**
 * @brief Smooth intersection
 */
inline double opSmoothIntersection(double d1, double d2, double k) {
    double h = std::clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h);
}

//==============================================================================
// DOMAIN OPERATIONS (Modify the input point before distance calculation)
//==============================================================================

/**
 * @brief Repeat space in a grid pattern
 */
inline Vec3 opRepeat(const Vec3& p, const Vec3& period) {
    return Vec3(
        p.x() - period.x() * std::round(p.x() / period.x()),
        p.y() - period.y() * std::round(p.y() / period.y()),
        p.z() - period.z() * std::round(p.z() / period.z())
    );
}

/**
 * @brief Twist the space around Y axis
 */
inline Vec3 opTwist(const Vec3& p, double amount) {
    double c = std::cos(amount * p.y());
    double s = std::sin(amount * p.y());
    return Vec3(
        c * p.x() - s * p.z(),
        p.y(),
        s * p.x() + c * p.z()
    );
}

//==============================================================================
// RAY MARCHING
//==============================================================================

/**
 * @brief Configuration for ray marching
 */
struct RayMarchConfig {
    int max_steps = 128;           // Maximum ray marching steps
    double epsilon = 0.0001;       // Surface hit threshold (smaller = more accurate)
    double max_distance = 100.0;   // Maximum ray travel distance
    double normal_epsilon = 0.0001; // Epsilon for normal estimation (smaller = better normals)
    
    RayMarchConfig() = default;
    
    RayMarchConfig(int steps, double eps, double max_dist)
        : max_steps(steps), epsilon(eps), max_distance(max_dist), normal_epsilon(eps) {}
};

/**
 * @brief Result of a ray marching operation
 */
struct RayMarchResult {
    bool hit;           // True if surface was hit
    double t;           // Distance along ray
    Point3 position;    // Hit position
    Vec3 normal;        // Surface normal
    int steps;          // Number of steps taken
    
    RayMarchResult() : hit(false), t(0.0), steps(0) {}
};

/**
 * @brief Estimate surface normal using gradient of distance field
 * @param p Point on surface
 * @param dist_func Distance function to evaluate
 * @param epsilon Step size for finite differences
 */
template<typename DistFunc>
inline Vec3 estimateNormal(const Point3& p, DistFunc dist_func, double epsilon = 0.001) {
    // Central differences for gradient
    Vec3 grad(
        dist_func(p + Vec3(epsilon, 0, 0)) - dist_func(p - Vec3(epsilon, 0, 0)),
        dist_func(p + Vec3(0, epsilon, 0)) - dist_func(p - Vec3(0, epsilon, 0)),
        dist_func(p + Vec3(0, 0, epsilon)) - dist_func(p - Vec3(0, 0, epsilon))
    );
    
    return unit_vector(grad);
}

/**
 * @brief Sphere tracing / ray marching algorithm
 * @param r Ray to march
 * @param dist_func Distance function to evaluate
 * @param ray_t Valid t interval
 * @param config Ray marching configuration
 * @return Ray march result with hit info
 */
template<typename DistFunc>
inline RayMarchResult rayMarch(const Ray& r, DistFunc dist_func, Interval ray_t, const RayMarchConfig& config = RayMarchConfig()) {
    RayMarchResult result;
    
    double t = ray_t.min;
    
    for (int step = 0; step < config.max_steps; step++) {
        result.steps = step + 1;
        
        // Current position along ray
        Point3 p = r.at(t);
        
        // Evaluate distance function
        double dist = dist_func(p);
        
        // Check if we hit the surface
        if (dist < config.epsilon) {
            result.hit = true;
            result.t = t;
            result.position = p;
            
            // Estimate normal first
            result.normal = estimateNormal(p, dist_func, config.normal_epsilon);
            
            // Optional: Push the hit point slightly along the normal to avoid self-intersection
            // result.position = p + result.normal * config.epsilon;
            
            return result;
        }
        
        // Advance ray by safe distance
        t += dist;
        
        // Check if we exceeded maximum distance
        if (t > ray_t.max || t > config.max_distance) {
            break;
        }
    }
    
    // No hit
    result.hit = false;
    return result;
}

} // namespace SDF
