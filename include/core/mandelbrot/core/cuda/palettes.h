#pragma once

#include "mandelbrot/core/cuda/color.h"
#include "mandelbrot/core/cuda/defines.h"

#include <cuda_runtime.h>

namespace mandelbrot {
namespace cuda {

__device__ static constexpr uint32_t blue_palette[] = {
    MakeRGB(30, 16, 48),    MakeRGB(15, 12, 94),    MakeRGB(3, 28, 150),
    MakeRGB(6, 50, 188),    MakeRGB(20, 80, 195),   MakeRGB(57, 125, 209),
    MakeRGB(73, 155, 216),  MakeRGB(134, 181, 229), MakeRGB(170, 210, 235),
    MakeRGB(240, 249, 250), MakeRGB(246, 231, 161), MakeRGB(252, 212, 70),
    MakeRGB(251, 175, 42),  MakeRGB(215, 126, 41),  MakeRGB(187, 79, 39),
    MakeRGB(110, 50, 53)};

__device__ static constexpr uint32_t pretty_palette[] = {
    MakeRGB(25, 7, 26),     MakeRGB(9, 17, 45),     MakeRGB(4, 31, 68),
    MakeRGB(12, 58, 108),   MakeRGB(26, 87, 163),   MakeRGB(56, 132, 190),
    MakeRGB(80, 170, 207),  MakeRGB(120, 190, 224), MakeRGB(159, 215, 243),
    MakeRGB(207, 233, 254), MakeRGB(255, 245, 232), MakeRGB(255, 215, 182),
    MakeRGB(254, 173, 122), MakeRGB(243, 112, 95),  MakeRGB(205, 53, 82),
    MakeRGB(128, 18, 70)};

__device__ static constexpr uint32_t artistic_palette[] = {
    MakeRGB(5, 4, 32),      MakeRGB(36, 5, 91),     MakeRGB(142, 13, 172),
    MakeRGB(203, 32, 198),  MakeRGB(251, 78, 168),  MakeRGB(255, 127, 127),
    MakeRGB(255, 188, 97),  MakeRGB(255, 221, 84),  MakeRGB(255, 252, 105),
    MakeRGB(224, 240, 141), MakeRGB(168, 227, 168), MakeRGB(85, 194, 193),
    MakeRGB(30, 129, 176),  MakeRGB(10, 76, 150),   MakeRGB(3, 35, 102),
    MakeRGB(1, 11, 54)};

__device__ static constexpr uint32_t natural_palette[] = {
    MakeRGB(23, 12, 30),    MakeRGB(54, 28, 53),    MakeRGB(98, 56, 67),
    MakeRGB(144, 95, 87),   MakeRGB(181, 128, 87),  MakeRGB(217, 155, 85),
    MakeRGB(237, 183, 105), MakeRGB(244, 202, 133), MakeRGB(244, 220, 170),
    MakeRGB(232, 232, 207), MakeRGB(210, 229, 186), MakeRGB(169, 213, 150),
    MakeRGB(106, 191, 139), MakeRGB(53, 158, 118),  MakeRGB(29, 105, 78),
    MakeRGB(18, 63, 52)};

__device__ static constexpr uint32_t gradient_palette[] = {
    MakeRGB(15, 32, 80),    MakeRGB(69, 117, 180),  MakeRGB(116, 173, 209),
    MakeRGB(171, 217, 233), MakeRGB(224, 243, 248), MakeRGB(253, 219, 199),
    MakeRGB(244, 165, 130), MakeRGB(214, 96, 77),   MakeRGB(178, 24, 43)};

__device__ static constexpr uint32_t cosmic_palette[] = {
    MakeRGB(12, 12, 35),    MakeRGB(23, 23, 50),    MakeRGB(45, 45, 90),
    MakeRGB(67, 67, 120),   MakeRGB(100, 100, 155), MakeRGB(140, 140, 190),
    MakeRGB(90, 140, 160),  MakeRGB(70, 180, 180),  MakeRGB(50, 220, 200),
    MakeRGB(30, 255, 220),  MakeRGB(20, 240, 210),  MakeRGB(10, 220, 200),
    MakeRGB(255, 180, 200), MakeRGB(255, 140, 180), MakeRGB(255, 100, 160),
    MakeRGB(255, 80, 130),  MakeRGB(255, 60, 100),  MakeRGB(255, 40, 80),
    MakeRGB(255, 20, 30),   MakeRGB(240, 10, 40),   MakeRGB(220, 0, 50),
    MakeRGB(200, 0, 60),    MakeRGB(180, 0, 70),    MakeRGB(160, 0, 80),
    MakeRGB(255, 255, 255), MakeRGB(255, 240, 180), MakeRGB(255, 230, 140),
    MakeRGB(255, 220, 100), MakeRGB(255, 200, 60),  MakeRGB(255, 180, 20)};

__device__ static constexpr uint32_t black_white_palette[] = {
    MakeRGB(0, 0, 0), MakeRGB(255, 255, 255)};

__device__ static constexpr size_t blue_palette_size =
    sizeof(blue_palette) / sizeof(uint32_t);

__device__ static constexpr size_t pretty_palette_size =
    sizeof(pretty_palette) / sizeof(uint32_t);

__device__ static constexpr size_t artistic_palette_size =
    sizeof(artistic_palette) / sizeof(uint32_t);

__device__ static constexpr size_t natural_palette_size =
    sizeof(natural_palette) / sizeof(uint32_t);

__device__ static constexpr size_t gradient_palette_size =
    sizeof(gradient_palette) / sizeof(uint32_t);

__device__ static constexpr size_t cosmic_palette_size =
    sizeof(cosmic_palette) / sizeof(uint32_t);

__device__ static constexpr size_t black_white_palette_size =
    sizeof(black_white_palette) / sizeof(uint32_t);

}  // namespace cuda
}  // namespace mandelbrot