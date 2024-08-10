#include "mandelbrot/core/render_options.h"

namespace mandelbrot {

bool operator==(const RenderOptions& left, const RenderOptions& right) {
  return left.coloring_mode == right.coloring_mode &&
         left.palette == right.palette &&
         left.max_iterations == right.max_iterations &&
         left.smoothing == right.smoothing;
}

bool operator!=(const RenderOptions& left, const RenderOptions& right) {
  return !(left == right);
}

}  // namespace mandelbrot