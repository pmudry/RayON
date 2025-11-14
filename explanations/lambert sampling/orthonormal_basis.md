# Computing tangent and bitangent vectors from surface normal

Given a unit normal vector $n$, you need to find two orthogonal vectors $t$ (tangent) and $b$ (bitangent) such that ${t, b, n}$ form **an orthonormal basis**.

## General Approach
The key is to pick an arbitrary vector that's not parallel to $n$, then use the Gram-Schmidt process or cross products:

- Choose a reference vector that's not parallel to $n$
- Compute the tangent using cross product
- Compute the bitangent using another cross product

```cpp
__device__ void build_orthonormal_basis(const float3& n, float3& tangent, float3& bitangent) {
    float3 up = fabs(n.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}
```

### Why `fabs()`, the absolute value ?

We use absolute value because `(-0.9, 0.1, 0.1)` and `(0.9, 0.1, 0.1)` should both pick the Y or Z axis as reference. The sign doesn't matter for determining which axis is most perpendicular.

## Key Points

- Non-uniqueness: There are infinitely many valid tangent/bitangent pairs (any rotation around the normal is valid)
- Handedness: The order of cross products determines if you get a right-handed or left-handed coordinate system
- Reference vector choice: Pick the axis that's most perpendicular to the normal to avoid numerical issues
