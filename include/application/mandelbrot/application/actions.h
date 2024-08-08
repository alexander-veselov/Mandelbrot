#pragma once

namespace mandelbrot {

enum class MouseAction {
  kOther,
  kPress,
  kRelease
};

enum class ScrollAction {
  kOther,
  kScrollUp,
  kScrollDown
};

using KeyAction = MouseAction;

}  // namespace mandelbrot