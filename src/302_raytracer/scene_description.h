/**
 * @file scene_description.h
 * @brief Unified scene description format for CPU and GPU renderers
 * 
 * This file provides a common scene representation that can be:
 * 1. Built once on the host
 * 2. Converted to CPU-friendly format (polymorphic, with virtual functions)
 * 3. Converted to GPU-friendly format (flat arrays, no virtual functions)
 * 
 * Supports: primitives, triangle meshes, BVH acceleration, ray marching (SDFs)
 */

#pragma once

#include "vec3.h"
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace Scene {

//==============================================================================
// MATERIAL SYSTEM
//==============================================================================

enum class MaterialType : uint8_t {
    LAMBERTIAN,
    METAL,
    MIRROR,
    ROUGH_MIRROR,
    GLASS,
    DIELECTRIC,
    LIGHT,
    CONSTANT,
    SHOW_NORMALS,
    SDF_MATERIAL     // For ray-marched objects
};

/**
 * @brief Procedural pattern types for material coloring
 */
enum class ProceduralPattern : uint8_t {
    NONE,                    // No pattern, use solid color
    FIBONACCI_DOTS,          // Regularly spaced dots using Fibonacci grid
    CHECKERBOARD,           // Checkerboard pattern
    STRIPES                 // Striped pattern
};

/**
 * @brief Material description - unified format for all material types
 */
struct MaterialDesc {
    MaterialType type;
    
    Vec3 albedo;              // Base color / reflectance
    Vec3 emission;            // Emissive color (for lights)
    float roughness;          // Surface roughness [0-1]
    float metallic;           // Metallic factor [0-1]
    float refractive_index;   // For glass/dielectric materials
    float transmission;       // Transparency [0-1]
    int texture_id;           // Texture index (-1 = none, for future use)
    
    // Procedural pattern support
    ProceduralPattern pattern;    // Pattern type
    Vec3 pattern_color;           // Secondary color for pattern
    float pattern_param1;         // Pattern-specific parameter 1 (e.g., dot count, scale)
    float pattern_param2;         // Pattern-specific parameter 2 (e.g., dot radius)
    
    // Default constructor
    MaterialDesc() 
        : type(MaterialType::LAMBERTIAN)
        , albedo(0.7, 0.7, 0.7)
        , emission(0, 0, 0)
        , roughness(0.0f)
        , metallic(0.0f)
        , refractive_index(1.0f)
        , transmission(0.0f)
        , texture_id(-1)
        , pattern(ProceduralPattern::NONE)
        , pattern_color(0, 0, 0)
        , pattern_param1(0.0f)
        , pattern_param2(0.0f)
    {}
    
    // Factory methods for common materials
    static MaterialDesc lambertian(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::LAMBERTIAN;
        mat.albedo = color;
        return mat;
    }
    
    static MaterialDesc metal(const Vec3& color, float roughness = 0.0f) {
        MaterialDesc mat;
        mat.type = MaterialType::METAL;
        mat.albedo = color;
        mat.roughness = roughness;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc mirror(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::MIRROR;
        mat.albedo = color;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc roughMirror(const Vec3& color, float roughness) {
        MaterialDesc mat;
        mat.type = MaterialType::ROUGH_MIRROR;
        mat.albedo = color;
        mat.roughness = roughness;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc glass(float ior) {
        MaterialDesc mat;
        mat.type = MaterialType::GLASS;
        mat.albedo = Vec3(1, 1, 1);
        mat.refractive_index = ior;
        mat.transmission = 1.0f;
        return mat;
    }
    
    static MaterialDesc light(const Vec3& emission) {
        MaterialDesc mat;
        mat.type = MaterialType::LIGHT;
        mat.emission = emission;
        mat.albedo = Vec3(1, 1, 1);
        return mat;
    }
    
    static MaterialDesc constant(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::CONSTANT;
        mat.albedo = color;
        return mat;
    }
    
    // Factory methods for patterned materials
    static MaterialDesc fibonacciDots(const Vec3& base_color, const Vec3& dot_color, 
                                      int dot_count = 12, float dot_radius = 0.33f) {
        MaterialDesc mat;
        mat.type = MaterialType::LAMBERTIAN;
        mat.albedo = base_color;
        mat.pattern = ProceduralPattern::FIBONACCI_DOTS;
        mat.pattern_color = dot_color;
        mat.pattern_param1 = static_cast<float>(dot_count);
        mat.pattern_param2 = dot_radius;
        return mat;
    }
};

//==============================================================================
// GEOMETRY SYSTEM
//==============================================================================

enum class GeometryType : uint8_t {
    SPHERE,
    RECTANGLE,
    CUBE,
    DISPLACED_SPHERE,    // Sphere with procedural displacement
    TRIANGLE,            // Single triangle
    TRIANGLE_MESH,       // Collection of triangles (uses BVH)
    SDF_PRIMITIVE        // Ray-marched signed distance field
};

/**
 * @brief Triangle for mesh representation
 */
struct Triangle {
    Vec3 v0, v1, v2;           // Vertices
    Vec3 n0, n1, n2;           // Vertex normals (for smooth shading)
    Vec3 uv0, uv1, uv2;        // Texture coordinates (future use)
    bool has_normals;          // True if normals are provided
    
    Triangle() : has_normals(false) {}
    
    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2)
        : v0(v0), v1(v1), v2(v2), has_normals(false) {}
};

/**
 * @brief Triangle mesh with optional BVH
 */
struct TriangleMesh {
    std::vector<Triangle> triangles;
    int bvh_root_id;           // Index into BVH node array (-1 = not built)
    Vec3 bounds_min, bounds_max;
    
    TriangleMesh() : bvh_root_id(-1) {}
    
    // Factory methods (to be implemented)
    // static TriangleMesh fromOBJ(const std::string& filename);
    // static TriangleMesh cube(const Vec3& center, double size);
    // static TriangleMesh sphere(const Vec3& center, double radius, int subdivisions);
};

/**
 * @brief Signed Distance Field primitives for ray marching
 */
enum class SDFType : uint8_t {
    SPHERE,
    BOX,
    TORUS,
    CAPSULE,
    MANDELBULB,
    CUSTOM         // User-defined distance function
};

struct SDFPrimitive {
    SDFType type;
    Vec3 position;
    Vec3 parameters;       // Size/shape parameters
    float blend_factor;    // For smooth blending operations
    int operation;         // Union, subtract, intersect
    
    SDFPrimitive() : type(SDFType::SPHERE), blend_factor(0.0f), operation(0) {}
};

/**
 * @brief Main geometry description - unified format for all geometry types
 */
struct GeometryDesc {
    GeometryType type;
    int material_id;       // Index into materials array
    
    // Geometry-specific data stored in union for memory efficiency
    union GeomData {
        // Sphere
        struct {
            Vec3 center;
            double radius;
        } sphere;
        
        // Rectangle (quad)
        struct {
            Vec3 corner;
            Vec3 u, v;      // Edge vectors
        } rectangle;
        
        // Cube
        struct {
            Vec3 center;
            double side_length;
            Vec3 rotation;  // Euler angles in degrees
        } cube;
        
        // Displaced sphere (golf ball, etc.)
        struct {
            Vec3 center;
            double radius;
            float displacement_scale;
            int pattern_type;
        } displaced_sphere;
        
        // Single triangle
        struct {
            Vec3 v0, v1, v2;
            Vec3 n0, n1, n2;
            bool has_normals;
        } triangle;
        
        // Mesh instance
        struct {
            int mesh_id;        // Index into mesh array
            Vec3 translation;
            Vec3 rotation;      // Euler angles
            Vec3 scale;
            bool use_bvh;       // Enable BVH acceleration for this instance
        } mesh_instance;
        
        // SDF primitive
        struct {
            SDFType sdf_type;
            Vec3 position;
            Vec3 parameters;
            float max_distance;  // Ray marching limit
            float epsilon;       // Surface precision
        } sdf;
        
        GeomData() { memset(this, 0, sizeof(GeomData)); }
    } data;
    
    // Axis-aligned bounding box (for BVH)
    Vec3 bounds_min, bounds_max;
    
    GeometryDesc() : type(GeometryType::SPHERE), material_id(0) {
        bounds_min = Vec3(-1, -1, -1);
        bounds_max = Vec3(1, 1, 1);
    }
};

//==============================================================================
// BVH ACCELERATION STRUCTURE (for future implementation)
//==============================================================================

struct BVHNode {
    Vec3 bounds_min, bounds_max;  // Axis-aligned bounding box
    
    // Interior node: left/right child indices
    // Leaf node: geometry indices
    union NodeData {
        struct {
            int left_child;
            int right_child;
        } interior;
        
        struct {
            int first_geom_idx;
            int geom_count;
        } leaf;
        
        NodeData() { memset(this, 0, sizeof(NodeData)); }
    } data;
    
    bool is_leaf;
    uint8_t split_axis;  // 0=x, 1=y, 2=z
    
    BVHNode() : is_leaf(true), split_axis(0) {}
};

struct BVHTree {
    std::vector<BVHNode> nodes;
    int root_index;
    
    BVHTree() : root_index(-1) {}
    
    // Build methods (to be implemented in Phase 5)
    // void build(const std::vector<GeometryDesc>& geometries);
    // void buildSAH(const std::vector<GeometryDesc>& geometries);  // Surface Area Heuristic
};

//==============================================================================
// COMPLETE SCENE DESCRIPTION
//==============================================================================

/**
 * @brief Complete scene description - single source of truth for all renderers
 */
class SceneDescription {
public:
    std::vector<MaterialDesc> materials;
    std::vector<GeometryDesc> geometries;
    std::vector<TriangleMesh> meshes;
    std::vector<BVHTree> bvh_trees;          // Per-mesh BVH
    BVHTree top_level_bvh;                    // Scene-level BVH
    
    // Scene properties
    Vec3 camera_position;
    Vec3 camera_look_at;
    Vec3 camera_up;
    float camera_fov;
    
    // Rendering settings
    Vec3 background_color;
    float ambient_light;
    bool use_bvh;                             // Enable scene BVH
    
    // Constructor
    SceneDescription() 
        : camera_position(0, 0, 0)
        , camera_look_at(0, 0, -1)
        , camera_up(0, 1, 0)
        , camera_fov(90.0f)
        , background_color(0.5f, 0.7f, 1.0f)
        , ambient_light(0.1f)
        , use_bvh(false)
    {}
    
    //==========================================================================
    // BUILDER INTERFACE
    //==========================================================================
    
    /**
     * @brief Add a material to the scene
     * @return Material ID (index in materials array)
     */
    int addMaterial(const MaterialDesc& mat) {
        materials.push_back(mat);
        return static_cast<int>(materials.size() - 1);
    }
    
    /**
     * @brief Add geometry to the scene
     */
    void addGeometry(const GeometryDesc& geom) {
        geometries.push_back(geom);
    }
    
    /**
     * @brief Add a triangle mesh to the scene
     * @return Mesh ID (index in meshes array)
     */
    int addMesh(const TriangleMesh& mesh) {
        meshes.push_back(mesh);
        return static_cast<int>(meshes.size() - 1);
    }
    
    //==========================================================================
    // HIGH-LEVEL API - Convenience methods
    //==========================================================================
    
    void addSphere(const Vec3& center, double radius, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::SPHERE;
        geom.material_id = mat_id;
        geom.data.sphere.center = center;
        geom.data.sphere.radius = radius;
        
        // Compute bounding box
        Vec3 r(radius, radius, radius);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addDisplacedSphere(const Vec3& center, double radius, int mat_id, 
                           float displacement_scale = 0.2f, int pattern_type = 0) {
        GeometryDesc geom;
        geom.type = GeometryType::DISPLACED_SPHERE;
        geom.material_id = mat_id;
        geom.data.displaced_sphere.center = center;
        geom.data.displaced_sphere.radius = radius;
        geom.data.displaced_sphere.displacement_scale = displacement_scale;
        geom.data.displaced_sphere.pattern_type = pattern_type; // 0 = golf ball dimples
        
        // Compute bounding box (slightly larger due to displacement)
        Vec3 r(radius * 1.1, radius * 1.1, radius * 1.1);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addRectangle(const Vec3& corner, const Vec3& u, const Vec3& v, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::RECTANGLE;
        geom.material_id = mat_id;
        geom.data.rectangle.corner = corner;
        geom.data.rectangle.u = u;
        geom.data.rectangle.v = v;
        
        // Compute bounding box (all 4 corners)
        Vec3 corners[4] = {
            corner,
            corner + u,
            corner + v,
            corner + u + v
        };
        geom.bounds_min = corners[0];
        geom.bounds_max = corners[0];
        for (int i = 1; i < 4; i++) {
            geom.bounds_min = Vec3(
                std::min(geom.bounds_min.x(), corners[i].x()),
                std::min(geom.bounds_min.y(), corners[i].y()),
                std::min(geom.bounds_min.z(), corners[i].z())
            );
            geom.bounds_max = Vec3(
                std::max(geom.bounds_max.x(), corners[i].x()),
                std::max(geom.bounds_max.y(), corners[i].y()),
                std::max(geom.bounds_max.z(), corners[i].z())
            );
        }
        
        geometries.push_back(geom);
    }
    
    void addDisplacedSphere(const Vec3& center, double radius, int mat_id, int pattern_type = 0) {
        GeometryDesc geom;
        geom.type = GeometryType::DISPLACED_SPHERE;
        geom.material_id = mat_id;
        geom.data.displaced_sphere.center = center;
        geom.data.displaced_sphere.radius = radius;
        geom.data.displaced_sphere.displacement_scale = 0.05f;  // Default
        geom.data.displaced_sphere.pattern_type = pattern_type;
        
        // Compute bounding box (slightly larger for displacement)
        Vec3 r(radius * 1.1, radius * 1.1, radius * 1.1);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::TRIANGLE;
        geom.material_id = mat_id;
        geom.data.triangle.v0 = v0;
        geom.data.triangle.v1 = v1;
        geom.data.triangle.v2 = v2;
        geom.data.triangle.has_normals = false;
        
        // Compute bounding box
        geom.bounds_min = Vec3(
            std::min(std::min(v0.x(), v1.x()), v2.x()),
            std::min(std::min(v0.y(), v1.y()), v2.y()),
            std::min(std::min(v0.z(), v1.z()), v2.z())
        );
        geom.bounds_max = Vec3(
            std::max(std::max(v0.x(), v1.x()), v2.x()),
            std::max(std::max(v0.y(), v1.y()), v2.y()),
            std::max(std::max(v0.z(), v1.z()), v2.z())
        );
        
        geometries.push_back(geom);
    }
    
    void addMeshInstance(int mesh_id, const Vec3& pos, const Vec3& rot, const Vec3& scale, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::TRIANGLE_MESH;
        geom.material_id = mat_id;
        geom.data.mesh_instance.mesh_id = mesh_id;
        geom.data.mesh_instance.translation = pos;
        geom.data.mesh_instance.rotation = rot;
        geom.data.mesh_instance.scale = scale;
        geom.data.mesh_instance.use_bvh = true;
        
        // TODO: Compute transformed bounding box
        geom.bounds_min = Vec3(-1, -1, -1);
        geom.bounds_max = Vec3(1, 1, 1);
        
        geometries.push_back(geom);
    }
    
    void addSDFPrimitive(SDFType type, const Vec3& pos, const Vec3& params, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::SDF_PRIMITIVE;
        geom.material_id = mat_id;
        geom.data.sdf.sdf_type = type;
        geom.data.sdf.position = pos;
        geom.data.sdf.parameters = params;
        geom.data.sdf.max_distance = 10.0f;  // Default ray march distance
        geom.data.sdf.epsilon = 0.001f;       // Default surface precision
        
        // TODO: Compute proper bounding box for SDF
        geom.bounds_min = pos - params;
        geom.bounds_max = pos + params;
        
        geometries.push_back(geom);
    }
    
    //==========================================================================
    // BVH CONSTRUCTION (Phase 5)
    //==========================================================================
    
    void buildBVH() {
        // To be implemented in Phase 5
        // This will build the top-level BVH for the scene
    }
    
    void buildMeshBVHs() {
        // To be implemented in Phase 5
        // This will build BVH for each mesh
    }
    
    //==========================================================================
    // SERIALIZATION (Future)
    //==========================================================================
    
    bool saveToFile(const std::string& filename) const {
        // To be implemented
        return false;
    }
    
    bool loadFromFile(const std::string& filename) {
        // To be implemented
        return false;
    }
    
    //==========================================================================
    // VALIDATION
    //==========================================================================
    
    bool validate() const {
        // Check material IDs are valid
        for (const auto& geom : geometries) {
            if (geom.material_id < 0 || geom.material_id >= static_cast<int>(materials.size())) {
                return false;
            }
        }
        
        // Check mesh IDs are valid
        for (const auto& geom : geometries) {
            if (geom.type == GeometryType::TRIANGLE_MESH) {
                int mesh_id = geom.data.mesh_instance.mesh_id;
                if (mesh_id < 0 || mesh_id >= static_cast<int>(meshes.size())) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

} // namespace Scene
