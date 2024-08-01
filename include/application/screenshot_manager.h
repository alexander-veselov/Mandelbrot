#pragma once

#include "complex.h"

namespace MandelbrotSet {

class ScreenshotManager {
 public:
  ScreenshotManager(const ScreenshotManager&) = delete;
  ScreenshotManager& operator= (const ScreenshotManager&) = delete;
  static const ScreenshotManager& Instance();

  static void MakeScreenshot();

 private:
  ScreenshotManager();
};

}