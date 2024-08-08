#pragma once

#include "mandelbrot/core/complex.h"

#include "mandelbrot/application/actions.h"

#include <atomic>

namespace mandelbrot {

class Explorer {
 public:
  Explorer(const Complex& position, double_t zoom);

  void MouseClickedEvent(const Complex& position);
  void MouseReleasedEvent(const Complex& position);
  void MouseMovedEvent(const Complex& position);
  void MouseScrollEvent(const Complex& position, ScrollAction action);

  void Navigate(const Complex& position, double_t zoom);

  Complex GetCenterPosition() const noexcept;
  Complex GetDisplayPosition() const noexcept;
  double_t GetZoom() const noexcept;

 private:
  Complex center_position_;
  Complex display_position_;
  Complex click_position_;
  double_t zoom_;
  std::atomic<bool> moving_;
};

}