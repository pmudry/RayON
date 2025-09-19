#pragma once

const double aspect_ratio = 16.0 / 9.0;

const int SET_IMAGE_HEIGHT = 720;

const int image_height = SET_IMAGE_HEIGHT;
const int image_width = (int)(aspect_ratio * SET_IMAGE_HEIGHT);
const int channels = 3; // RGB

// Renderer specific settings
int samples_per_pixel = 16; // Number of samples per pixel for anti-aliasing
int max_depth = 16; // Maximum recursion depth for ray tracing
