#pragma once

#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/point.h"
#include "mandelbrot/core/size.h"
#include "mandelbrot/core/typedefs.h"
#include "mandelbrot/application/actions.h"

#include <atomic>

namespace mandelbrot {

  class Explorer {
  public:
    Explorer(const Complex& position, double_t zoom);

    void MouseClickedEvent(const Point& position);
    void MouseReleasedEvent(const Point& position);
    void MouseMovedEvent(const Point& position);
    void MouseScrollEvent(const Point& position, ScrollAction action);

    void Navigate(const Complex& position, double_t zoom);
    void Chop();

    Complex GetReferenceCenter() const noexcept;
    Complex GetOffset() const noexcept;
    Complex GetCenterPosition() const noexcept;
    double_t GetZoom() const noexcept;

  private:
    Complex ScreenToWorld(const Point& p) const noexcept;

  private:
    Complex reference_center_;
    Complex dc_;

    Point click_position_;
    Complex drag_start_dc_;

    Size screen_size_ = Size{ 1024, 768 };

    double_t zoom_;
    std::atomic<bool> moving_;
  };

}  // namespace mandelbrot