#pragma once

#include "complex.h"

#include <atomic>

namespace MandelbrotSet {

// TODO: make not global
constexpr auto kZoomFactor = 1.5;
constexpr auto kDirectionalZoom = true;

class Explorer {
 public:
  enum class ScrollEvent { kScrollUp, kScrollDown };

  Explorer(const Complex& position, double_t zoom);

  void MouseClickedEvent(const Complex& position);
  void MouseReleasedEvent(const Complex& position);
  void MouseMovedEvent(const Complex& position);
  void MouseScrollEvent(const Complex& position, ScrollEvent event);

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