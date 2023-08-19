#pragma once

__global__ void MandelbrotSet(uint8_t* data, uint32_t width, uint32_t height,
                              double_t center_real, double_t center_imag,
                              double_t zoom_factor);