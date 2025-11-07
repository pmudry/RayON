/**
 * @file sdf_shape.h
 * @brief Hittable wrapper for SDF primitives
 * 
 * This file provides a bridge between the SDF system and the traditional
 * ray tracing interface. SDFShape wraps various SDF primitives and makes
 * them compatible with the Hittable interface used by the CPU renderer.
 */

#pragma once

#include "hittable.h"
#include "sdf.h"
#include "material.h"
#include "utils.h"
#include "scene_description.h"
#include <memory>
#include <functional>

// Use SDFType from scene_description.h
using Scene::SDFType;

/**
 * @brief Hittable wrapper for SDF primitives
 * 
 * This class adapts SDF distance functions to work with the standard
 * ray tracing pipeline. It uses sphere tracing (ray marching) to find
 * intersections and estimates normals using the gradient method.
 */
class SDFShape : public Hittable {
public:
    /**
     * @brief Constructor for SDFShape
     */
    SDFShape(shared_ptr<Material> mat) : mat(mat) {
        // Initialize with a default distance function that returns a large distance
        distance_func = [](const Vec3& p) { return 1e10; };
    }
    
    /**
     * @brief Construct an SDF sphere
     */
    static shared_ptr<SDFShape> createSphere(const Vec3& center, double radius, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::SPHERE;
        shape->center = center;
        shape->param1 = radius;
        shape->rotation = rotation;
        
        // Distance function (spheres are rotationally symmetric, so rotation has no effect)
        shape->distance_func = [center, radius](const Vec3& p) {
            return SDF::sdSphere(p, center, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF box
     */
    static shared_ptr<SDFShape> createBox(const Vec3& center, const Vec3& size, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::BOX;
        shape->center = center;
        shape->size = size;
        shape->rotation = rotation;
        
        // Distance function with rotation support
        shape->distance_func = [center, size, rotation](const Vec3& p) {
            // Transform point into box's local space (apply inverse rotation)
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdBox(local_p, center, size);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF torus
     */
    static shared_ptr<SDFShape> createTorus(const Vec3& center, double major_radius, double minor_radius, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::TORUS;
        shape->center = center;
        shape->param1 = major_radius;
        shape->param2 = minor_radius;
        shape->rotation = rotation;
        
        // Distance function with rotation support
        shape->distance_func = [center, major_radius, minor_radius, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdTorus(local_p, center, major_radius, minor_radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF capsule
     */
    static shared_ptr<SDFShape> createCapsule(const Vec3& a, const Vec3& b, double radius, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CAPSULE;
        shape->center = a;
        shape->direction = b;
        shape->param1 = radius;
        shape->rotation = rotation;
        
        // Distance function with rotation
        Vec3 center = (a + b) * 0.5;
        shape->distance_func = [a, b, radius, rotation, center](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdCapsule(local_p, a, b, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF cylinder
     */
    static shared_ptr<SDFShape> createCylinder(const Vec3& center, double height, double radius, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CYLINDER;
        shape->center = center;
        shape->param1 = height;
        shape->param2 = radius;
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, height, radius, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdCylinder(local_p, center, height, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF plane
     */
    static shared_ptr<SDFShape> createPlane(const Vec3& normal, double distance, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::PLANE;
        shape->direction = unit_vector(normal);
        shape->param1 = distance;
        shape->rotation = rotation;
        
        // Distance function with rotation
        Vec3 n = unit_vector(normal);
        shape->distance_func = [n, distance, rotation](const Vec3& p) {
            // Rotate the normal instead of the point for planes
            Vec3 rotated_normal = SDF::applyRotation(n, rotation);
            return SDF::sdPlane(p, rotated_normal, distance);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Mandelbulb fractal
     */
    static shared_ptr<SDFShape> createMandelbulb(const Vec3& center, double power, int iterations, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::MANDELBULB;
        shape->center = center;
        shape->param1 = power;
        shape->param2 = static_cast<double>(iterations);
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, power, iterations, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdMandelbulb(local_p, center, power, iterations);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Death Star
     */
    static shared_ptr<SDFShape> createDeathStar(const Vec3& center, double main_radius, double cutout_radius, double cutout_distance, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::DEATH_STAR;
        shape->center = center;
        shape->param1 = main_radius;
        shape->param2 = cutout_radius;
        shape->size = Vec3(cutout_distance, 0, 0);  // Store cutout_distance in size.x
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, main_radius, cutout_radius, cutout_distance, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdDeathStar(local_p, center, main_radius, cutout_radius, cutout_distance);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Cut Hollow Sphere
     */
    static shared_ptr<SDFShape> createCutHollowSphere(const Vec3& center, double radius, double cut_height, double thickness, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CUT_HOLLOW_SPHERE;
        shape->center = center;
        shape->param1 = radius;
        shape->param2 = cut_height;
        shape->size = Vec3(thickness, 0, 0);  // Store thickness in size.x
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, radius, cut_height, thickness, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdCutHollowSphere(local_p, center, radius, cut_height, thickness);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Octahedron
     */
    static shared_ptr<SDFShape> createOctahedron(const Vec3& center, double size, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::OCTAHEDRON;
        shape->center = center;
        shape->param1 = size;
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, size, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdOctahedron(local_p, center, size);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Pyramid
     */
    static shared_ptr<SDFShape> createPyramid(const Vec3& center, double height, shared_ptr<Material> mat, const Vec3& rotation = Vec3(0, 0, 0)) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::PYRAMID;
        shape->center = center;
        shape->param1 = height;
        shape->rotation = rotation;
        
        // Distance function with rotation
        shape->distance_func = [center, height, rotation](const Vec3& p) {
            Vec3 local_p = SDF::applyInverseRotation(p - center, rotation) + center;
            return SDF::sdPyramid(local_p, center, height);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct a custom SDF with user-defined distance function
     */
    static shared_ptr<SDFShape> createCustom(std::function<double(const Vec3&)> dist_func, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CUSTOM;
        shape->distance_func = dist_func;
        return shape;
    }
    
    /**
     * @brief Ray-surface intersection test using sphere tracing
     */
    bool hit(const Ray& r, Interval ray_t, Hit_record& rec) const override {
        // Safety check
        if (!distance_func || !mat) {
            return false;
        }
        
        // Perform ray marching
        SDF::RayMarchResult result = SDF::rayMarch(r, distance_func, ray_t, config);
        
        if (result.hit) {
            rec.t = result.t;
            rec.p = result.position;
            rec.mat_ptr = mat;
            
            // Normal is already computed by ray marching
            rec.set_face_normal(r, result.normal);
            
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Set ray marching configuration
     */
    void setRayMarchConfig(const SDF::RayMarchConfig& cfg) {
        config = cfg;
    }
    
    /**
     * @brief Get the SDF type
     */
    SDFType getType() const { return type; }
    
    /**
     * @brief Get the center position (if applicable)
     */
    Vec3 getCenter() const { return center; }

private:
    SDFType type;
    shared_ptr<Material> mat;
    std::function<double(const Vec3&)> distance_func;
    
    // Shape parameters (meaning varies by type)
    Vec3 center{0, 0, 0};
    Vec3 size{1, 1, 1};
    Vec3 direction{0, 1, 0};
    Vec3 rotation{0, 0, 0};  // Rotation in radians (X, Y, Z)
    double param1 = 1.0;
    double param2 = 1.0;
    
    // Ray marching configuration
    SDF::RayMarchConfig config;
};

/**
 * @brief Helper class for combining multiple SDFs with operations
 */
class SDFCompound : public Hittable {
public:
    enum class Operation {
        UNION,
        SUBTRACTION,
        INTERSECTION,
        SMOOTH_UNION,
        SMOOTH_SUBTRACTION,
        SMOOTH_INTERSECTION
    };
    
    SDFCompound(shared_ptr<SDFShape> shape1, shared_ptr<SDFShape> shape2, 
                Operation op, shared_ptr<Material> mat, double blend = 0.5)
        : shape1(shape1), shape2(shape2), op(op), mat(mat), blend_factor(blend) {}
    
    bool hit(const Ray& r, Interval ray_t, Hit_record& rec) const override {
        // Create combined distance function based on operation
        auto combined_func = [this](const Vec3& p) -> double {
            // This is a simplified approach - we'd need to access the distance functions
            // from shape1 and shape2, which requires exposing them or storing them differently
            // For now, this is a placeholder structure
            return 0.0;  // TODO: Implement proper SDF combination
        };
        
        // Perform ray marching on combined SDF
        SDF::RayMarchConfig config;
        SDF::RayMarchResult result = SDF::rayMarch(r, combined_func, ray_t, config);
        
        if (result.hit) {
            rec.t = result.t;
            rec.p = result.position;
            rec.mat_ptr = mat;
            rec.set_face_normal(r, result.normal);
            return true;
        }
        
        return false;
    }
    
private:
    shared_ptr<SDFShape> shape1;
    shared_ptr<SDFShape> shape2;
    Operation op;
    shared_ptr<Material> mat;
    double blend_factor;
};
