#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

   // Host function declaration for tile-based CUDA rendering (for real-time display)
   unsigned long long renderPixelsCUDA(unsigned char *image, int width, int height, double cam_center_x,
                                       double cam_center_y, double cam_center_z, double pixel00_x, double pixel00_y,
                                       double pixel00_z, double delta_u_x, double delta_u_y, double delta_u_z,
                                       double delta_v_x, double delta_v_y, double delta_v_z, int samples_per_pixel,
                                       int max_depth);

   // Host function for accumulative rendering (adds samples to existing buffer)
   // d_rand_states: Pass nullptr to initialize, or pass existing device pointer to reuse
   unsigned long long renderPixelsCUDAAccumulative(unsigned char *image, float *accum_buffer,
                                                    int width, int height,
                                                    double cam_center_x, double cam_center_y, double cam_center_z,
                                                    double pixel00_x, double pixel00_y, double pixel00_z,
                                                    double delta_u_x, double delta_u_y, double delta_u_z,
                                                    double delta_v_x, double delta_v_y, double delta_v_z,
                                                    int samples_to_add, int total_samples_so_far, int max_depth,
                                                    void **d_rand_states);
   
   // Helper to free device random states
   void freeDeviceRandomStates(void *d_rand_states);
   
   // Set global light intensity (affects area light emission)
   void setLightIntensity(float intensity);
   
   // Set background gradient intensity (sky brightness)
   void setBackgroundIntensity(float intensity);

#ifdef __cplusplus
}
#endif