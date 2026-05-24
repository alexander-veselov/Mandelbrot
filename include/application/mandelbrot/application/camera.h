#pragma once

#include "mandelbrot/core/float.h"
#include "mandelbrot/core/complex.h"

namespace mandelbrot {

class Camera {
 public:
  Camera(const Complex& position, Float zoom);
  void Update(const Complex& position, Float zoom, Float dt);
  Complex GetPosition() const;
  Float GetZoom() const;

 private:
  Complex target_position_;
  Float target_zoom_;
  Complex current_position_;
  Float current_zoom_;
};

}  // namespace mandelbrot