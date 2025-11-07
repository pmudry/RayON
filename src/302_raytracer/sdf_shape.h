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
    static shared_ptr<SDFShape> createSphere(const Vec3& center, double radius, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::SPHERE;
        shape->center = center;
        shape->param1 = radius;
        
        // Distance function
        shape->distance_func = [center, radius](const Vec3& p) {
            return SDF::sdSphere(p, center, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF box
     */
    static shared_ptr<SDFShape> createBox(const Vec3& center, const Vec3& size, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::BOX;
        shape->center = center;
        shape->size = size;
        
        // Distance function
        shape->distance_func = [center, size](const Vec3& p) {
            return SDF::sdBox(p, center, size);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF torus
     */
    static shared_ptr<SDFShape> createTorus(const Vec3& center, double major_radius, double minor_radius, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::TORUS;
        shape->center = center;
        shape->param1 = major_radius;
        shape->param2 = minor_radius;
        
        // Distance function
        shape->distance_func = [center, major_radius, minor_radius](const Vec3& p) {
            return SDF::sdTorus(p, center, major_radius, minor_radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF capsule
     */
    static shared_ptr<SDFShape> createCapsule(const Vec3& a, const Vec3& b, double radius, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CAPSULE;
        shape->center = a;
        shape->direction = b;
        shape->param1 = radius;
        
        // Distance function
        shape->distance_func = [a, b, radius](const Vec3& p) {
            return SDF::sdCapsule(p, a, b, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF cylinder
     */
    static shared_ptr<SDFShape> createCylinder(const Vec3& center, double height, double radius, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::CYLINDER;
        shape->center = center;
        shape->param1 = height;
        shape->param2 = radius;
        
        // Distance function
        shape->distance_func = [center, height, radius](const Vec3& p) {
            return SDF::sdCylinder(p, center, height, radius);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF plane
     */
    static shared_ptr<SDFShape> createPlane(const Vec3& normal, double distance, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::PLANE;
        shape->direction = unit_vector(normal);
        shape->param1 = distance;
        
        // Distance function
        Vec3 n = unit_vector(normal);
        shape->distance_func = [n, distance](const Vec3& p) {
            return SDF::sdPlane(p, n, distance);
        };
        
        return shape;
    }
    
    /**
     * @brief Construct an SDF Mandelbulb fractal
     */
    static shared_ptr<SDFShape> createMandelbulb(const Vec3& center, double power, int iterations, shared_ptr<Material> mat) {
        auto shape = make_shared<SDFShape>(mat);
        shape->type = SDFType::MANDELBULB;
        shape->center = center;
        shape->param1 = power;
        shape->param2 = static_cast<double>(iterations);
        
        // Distance function
        shape->distance_func = [center, power, iterations](const Vec3& p) {
            return SDF::sdMandelbulb(p, center, power, iterations);
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
