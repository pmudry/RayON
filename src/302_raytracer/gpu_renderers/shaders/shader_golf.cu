#include "shader_golf.cuh"
#include "shader_common.cuh"

__device__ float3_simple fibonacci_point(int i, int n)
{
   const float ga = 2.39996323f;
   float k = (float)i + 0.5f;
   float v = k / (float)n;
   float phi = acosf(1.0f - 2.0f * v);
   float theta = ga * k;
   float s = sinf(phi);
   return float3_simple(cosf(theta) * s, sinf(theta) * s, cosf(phi));
}

__device__ float distanceToNearestDimple(float3_simple p)
{
   float3_simple q = unit_vector(p);
   const int N = 150;
   float max_dot = -1.0f;
   for (int i = 0; i < N; ++i)
   {
      float3_simple c = fibonacci_point(i, N);
      float d = dot(q, c);
      if (d > max_dot)
         max_dot = d;
   }
   max_dot = fmaxf(fminf(max_dot, 1.0f), -1.0f);
   return acosf(max_dot);
}


__device__ float hexagonalDimplePattern(float3_simple p)
{
   float ang = distanceToNearestDimple(unit_vector(p));
   const float dimple_radius = 0.24f;
   const float dimple_depth = 0.35f;
   if (ang < dimple_radius)
   {
      float t = ang / dimple_radius;
      float depth = dimple_depth * cosf(t * M_PI * 0.5f);
      return -depth;
   }
   return 0.0f;
}

__device__ float golfBallDisplacement(float3_simple p, float3_simple center, float radius)
{
   float3_simple local_point = float3_simple(p.x - center.x, p.y - center.y, p.z - center.z);
   float3_simple normalized = unit_vector(local_point);
   float displacement = hexagonalDimplePattern(normalized);
   return displacement;
}

__device__ bool hit_golf_ball_sphere(float3_simple center, float radius, const ray_simple &r, float t_min,
                                     float t_max, hit_record_simple &rec)
{
   if (!hit_sphere(center, radius, r, t_min, t_max, rec))
   {
      return false;
   }

   float3_simple surface_point = rec.p;
   float base_displacement = golfBallDisplacement(surface_point, center, radius);

   const float displacement_scale = 0.2f;
   const float dimple_depth_param = 0.35f;
   const float geo_strength = 0.35f;

   float3_simple base_outward =
       unit_vector(float3_simple(surface_point.x - center.x, surface_point.y - center.y, surface_point.z - center.z));

   float d_norm = fminf(1.0f, fmaxf(0.0f, -base_displacement / dimple_depth_param));
   float outward_push = radius * geo_strength * (1.0f - d_norm);
   rec.p = float3_simple(surface_point.x + base_outward.x * outward_push, surface_point.y + base_outward.y * outward_push,
                         surface_point.z + base_outward.z * outward_push);

   float3_simple base_normal = unit_vector(float3_simple(rec.p.x - center.x, rec.p.y - center.y, rec.p.z - center.z));

   if (base_displacement < -0.001f)
   {
      float3_simple helper = fabsf(base_normal.x) > 0.8f ? float3_simple(0, 1, 0) : float3_simple(1, 0, 0);
      float3_simple t1 = unit_vector(cross(helper, base_normal));
      float3_simple t2 = cross(base_normal, t1);

      const float h = 0.015f;
      float3_simple p_hat = base_normal;
      float d0 = hexagonalDimplePattern(p_hat);
      float d1 = hexagonalDimplePattern(unit_vector(float3_simple(p_hat.x + h * t1.x, p_hat.y + h * t1.y, p_hat.z + h * t1.z)));
      float d2 = hexagonalDimplePattern(unit_vector(float3_simple(p_hat.x + h * t2.x, p_hat.y + h * t2.y, p_hat.z + h * t2.z)));

      float dd1 = (d1 - d0) / h;
      float dd2 = (d2 - d0) / h;
      float3_simple grad_tan = float3_simple(dd1 * t1.x + dd2 * t2.x, dd1 * t1.y + dd2 * t2.y, dd1 * t1.z + dd2 * t2.z);

      float3_simple delta_n = float3_simple(-displacement_scale * grad_tan.x, -displacement_scale * grad_tan.y,
                                            -displacement_scale * grad_tan.z);

      float3_simple view_dir = unit_vector(float3_simple(-r.dir.x, -r.dir.y, -r.dir.z));
      float ndv = fmaxf(0.0f, dot(base_normal, view_dir));
      float atten = smoothstep(0.1f, 0.4f, ndv);
      delta_n = float3_simple(delta_n.x * atten, delta_n.y * atten, delta_n.z * atten);

      float max_len = 0.4f;
      float len = delta_n.length();
      if (len > max_len && len > 1e-6f)
      {
         delta_n = (max_len / len) * delta_n;
      }

      float3_simple perturbed = unit_vector(float3_simple(base_normal.x + delta_n.x, base_normal.y + delta_n.y, base_normal.z + delta_n.z));
      if (dot(perturbed, base_normal) < 0.0f)
      {
         perturbed = float3_simple(-perturbed.x, -perturbed.y, -perturbed.z);
      }
      if (!(perturbed.x == perturbed.x) || !(perturbed.y == perturbed.y) || !(perturbed.z == perturbed.z))
      {
         perturbed = base_normal;
      }
      rec.normal = perturbed;
   }
   else
   {
      rec.normal = base_normal;
   }

   float ndotv = dot(r.dir, rec.normal);
   if (!(ndotv == ndotv))
   {
      rec.normal = base_normal;
      ndotv = dot(r.dir, rec.normal);
   }
   rec.front_face = ndotv < 0;
   if (!rec.front_face)
   {
     rec.normal = float3_simple(-rec.normal.x, -rec.normal.y, -rec.normal.z);
   }

   const float surface_epsilon = 1e-3f;
   rec.p = float3_simple(rec.p.x + rec.normal.x * surface_epsilon, rec.p.y + rec.normal.y * surface_epsilon,
                         rec.p.z + rec.normal.z * surface_epsilon);
   return true;
}
