#pragma once

namespace MandelbrotSet {

enum class MouseAction {
  kPress,
  kRelease
};

enum class ScrollAction {
  kScrollUp,
  kScrollDown
};

using KeyAction = MouseAction;

}  // namespace MandelbrotSet