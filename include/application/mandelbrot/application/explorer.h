#pragma once

#include "mandelbrot/core/float.h"
#include "mandelbrot/core/complex.h"

#include "mandelbrot/application/actions.h"

#include <atomic>

namespace mandelbrot {

class Explorer {
 public:
  Explorer(const Complex& position, Float zoom);

  void MouseClickedEvent(const Complex& position);
  void MouseReleasedEvent(const Complex& position);
  void MouseMovedEvent(const Complex& position);
  void MouseScrollEvent(const Complex& position, ScrollAction action);

  void Navigate(const Complex& position, Float zoom);

  Complex GetCenterPosition() const noexcept;
  Complex GetDisplayPosition() const noexcept;
  Float GetZoom() const noexcept;

 private:
  Complex center_position_;
  Complex display_position_;
  Complex click_position_;
  Float zoom_;
  std::atomic<bool> moving_;
};

}