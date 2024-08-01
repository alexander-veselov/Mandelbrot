#include "screenshot_manager.h"

namespace MandelbrotSet {

ScreenshotManager::ScreenshotManager() {

}

const ScreenshotManager& ScreenshotManager::Instance() {
  static auto instance = ScreenshotManager{};
  return instance;
}

void ScreenshotManager::MakeScreenshot() {

}

}