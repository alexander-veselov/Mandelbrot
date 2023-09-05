#include "explorer.h"

namespace MandelbrotSet {

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
  center_position_.real += click_position_.real - position.real;
  center_position_.imag -= click_position_.imag - position.imag;
}

void Explorer::MouseMovedEvent(const Complex& position) {
  if (moving_) {
    display_position_.real =
        center_position_.real + (click_position_.real - position.real);
    display_position_.imag =
        center_position_.imag - (click_position_.imag - position.imag);
  }
}

void Explorer::MouseScrollEvent(ScrollEvent event) {

  // TODO: make kZoomFactor as constructor parameter
  constexpr static auto kZoomFactor = 1.5;

  if (event == ScrollEvent::kScrollUp) {
    zoom_ *= kZoomFactor;
  }

  if (event == ScrollEvent::kScrollDown) {
    zoom_ /= kZoomFactor;
  }
}

Complex Explorer::GetCenterPosition() const noexcept {
  return center_position_;
}

Complex Explorer::GetDisplayPosition() const noexcept {
  return display_position_;
}

double_t Explorer::GetZoom() const noexcept { return zoom_; }

}  // namespace MandelbrotSet