#pragma once

void MandelbrotSet(uint8_t* image, uint32_t image_width, uint32_t image_height,
                   double_t center_real, double_t center_imag,
                   double_t zoom_factor);

void MandelbrotSet(uint32_t* image, uint32_t image_width, uint32_t image_height,
                   double_t center_real, double_t center_imag,
                   double_t zoom_factor);