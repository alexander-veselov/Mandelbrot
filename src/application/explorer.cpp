#include "mandelbrot/application/explorer.h"

#include "mandelbrot/application/config.h"

namespace mandelbrot {

Explorer::Explorer(const Complex& position, double_t zoom)
    : center_position_{position},
      display_position_{position},
      click_position_{},
      zoom_{zoom},
      moving_{false} {}

void Explorer::MouseClickedEvent(const Complex& position) {
  moving_ = true;
  click_position_ = position;
}

void Explorer::MouseReleasedEvent(const Complex& position) {
  moving_ = false;
  center_position_ = display_position_;
}

void Explorer::MouseMovedEvent(const Complex& position) {
  if (moving_) {
    display_position_.real =
        center_position_.real + (click_position_.real - position.real);
    display_position_.imag =
        center_position_.imag + (click_position_.imag - position.imag);
  }
}

void Explorer::MouseScrollEvent(const Complex& position, ScrollAction action) {
  auto zoom_change = zoom_;
  const auto zoom_factor = GetConfig().zoom_factor;
  if (action == ScrollAction::kScrollUp) {
    zoom_change = zoom_factor;
  } else if (action == ScrollAction::kScrollDown) {
    zoom_change = 1. / zoom_factor;
  }

  zoom_ *= zoom_change;

  if (GetConfig().directional_zoom) {
    center_position_.real = position.real + (center_position_.real - position.real) / zoom_change;
    center_position_.imag = position.imag + (center_position_.imag - position.imag) / zoom_change;
    display_position_ = center_position_;
  }
}

Complex Explorer::GetCenterPosition() const noexcept {
  return center_position_;
}

Complex Explorer::GetDisplayPosition() const noexcept {
  return display_position_;
}

double_t Explorer::GetZoom() const noexcept {
  return zoom_;
}

}  // namespace mandelbrot