#pragma once

namespace mandelbrot {

enum class MouseAction {
  kPress,
  kRelease
};

enum class ScrollAction {
  kScrollUp,
  kScrollDown
};

using KeyAction = MouseAction;

}  // namespace mandelbrot