#include "mandelbrot/application/camera.h"

#include "mandelbrot/core/cuda/mandelbrot_set.h"

namespace mandelbrot {

static Float Lerp(Float a, Float b, Float t)
{
  return a * (1.0 - t) + (b * t);
}

Camera::Camera(const Complex& position, Float zoom)
    : target_position_ {position}, target_zoom_ {zoom},
      current_position_{position}, current_zoom_{zoom} {}

void Camera::Update(const Complex& position, Float zoom, Float dt) {
  target_position_ = position;
  target_zoom_ = zoom;

  const auto speed = Float{5};
  current_zoom_ = Lerp(current_zoom_, target_zoom_, dt * speed);
  current_position_.real = Lerp(current_position_.real, target_position_.real, dt * speed);
  current_position_.imag = Lerp(current_position_.imag, target_position_.imag, dt * speed);
}

Complex Camera::GetPosition() const {
  return current_position_;
}

Float Camera::GetZoom() const {
  return current_zoom_;
}

}  // namespace mandelbrot