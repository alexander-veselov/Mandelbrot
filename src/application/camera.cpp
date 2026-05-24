#include "mandelbrot/application/camera.h"

#include "mandelbrot/core/cuda/mandelbrot_set.h"

namespace mandelbrot {

static Float ExpSmooth(Float c, Float t, Float k) {
  return t + (c - t) * exp(-k);
}

Camera::Camera(const Complex& position, Float zoom)
    : target_position_ {position}, target_zoom_ {zoom},
      current_position_{position}, current_zoom_{zoom} {}

void Camera::Update(const Complex& position, Float zoom, Float dt) {
  target_position_ = position;
  target_zoom_ = zoom;

  const auto speed = Float{12.0};

  const auto previous_zoom = current_zoom_;
  current_zoom_ = ExpSmooth(current_zoom_, target_zoom_, dt * speed);

  const auto zoom_ratio = current_zoom_ / previous_zoom;

  current_position_.real = target_position_.real + (current_position_.real - target_position_.real) / zoom_ratio;
  current_position_.imag = target_position_.imag + (current_position_.imag - target_position_.imag) / zoom_ratio;

  current_position_.real = ExpSmooth(current_position_.real, target_position_.real, dt * speed);
  current_position_.imag = ExpSmooth(current_position_.imag, target_position_.imag, dt * speed);
}

Complex Camera::GetPosition() const {
  return current_position_;
}

Float Camera::GetZoom() const {
  return current_zoom_;
}

}  // namespace mandelbrot