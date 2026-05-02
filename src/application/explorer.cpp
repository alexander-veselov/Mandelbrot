#include "mandelbrot/application/explorer.h"

#include "mandelbrot/application/config.h"

namespace mandelbrot {

  namespace {
    constexpr double kWidth = 3.0;
    constexpr double kHeight = 2.0;
  }

  Explorer::Explorer(const Complex& position, double_t zoom)
    : reference_center_{ position },
    dc_{},
    click_position_{},
    drag_start_dc_{},
    zoom_{ zoom },
    moving_{ false } {
  }

  Complex Explorer::ScreenToWorld(const Point& p) const noexcept {
    const auto scale = 1. / std::min(screen_size_.width / kWidth,
      screen_size_.height / kHeight);

    const auto real = (p.x - Point::value_type{ screen_size_.width / 2. }) * Point::value_type{ scale };
    const auto imag = (p.y - Point::value_type{ screen_size_.height / 2. }) * Point::value_type{ scale };

    Complex out;
    out.real = real / Complex::value_type{ zoom_ };
    out.imag = imag / Complex::value_type{ zoom_ };
    return out;
  }

  void Explorer::MouseClickedEvent(const Point& position) {
    moving_ = true;
    click_position_ = position;
    drag_start_dc_ = dc_;
  }

  void Explorer::MouseReleasedEvent(const Point&) {
    moving_ = false;

    const auto threshold = Complex::value_type{ 1e-6 };

    if (abs(dc_.real) > threshold || abs(dc_.imag) > threshold) {
      reference_center_.real = reference_center_.real + dc_.real;
      reference_center_.imag = reference_center_.imag + dc_.imag;
      dc_ = {};
    }
  }

  void Explorer::Chop() {
      dc_.real =dc_.real / Complex::value_type{2.0};
      dc_.imag =dc_.imag / Complex::value_type{2.0};
      reference_center_.real = reference_center_.real + dc_.real;
      reference_center_.imag = reference_center_.imag + dc_.imag;
  }

  void Explorer::MouseMovedEvent(const Point& position) {
    if (!moving_) return;

    const auto start = ScreenToWorld(click_position_);
    const auto current = ScreenToWorld(position);

    dc_.real = drag_start_dc_.real + (start.real - current.real);
    dc_.imag = drag_start_dc_.imag - (start.imag - current.imag);
  }

  void Explorer::MouseScrollEvent(const Point& position, ScrollAction action) {
    const auto zoom_factor = GetConfig().zoom_factor;

    double_t factor = 1.0;
    if (action == ScrollAction::kScrollUp) {
      factor = zoom_factor;
    }
    else if (action == ScrollAction::kScrollDown) {
      factor = 1.0 / zoom_factor;
    }

    zoom_ *= factor;
  }

  void Explorer::Navigate(const Complex& position, double_t zoom) {
    reference_center_ = position;
    dc_ = {};
    zoom_ = zoom;
  }

  Complex Explorer::GetReferenceCenter() const noexcept {
    return reference_center_;
  }

  Complex Explorer::GetOffset() const noexcept {
    return dc_;
  }

  Complex Explorer::GetCenterPosition() const noexcept {
    return { reference_center_.real + dc_.real,
            reference_center_.imag + dc_.imag };
  }

  double_t Explorer::GetZoom() const noexcept {
    return zoom_;
  }

}  // namespace mandelbrot