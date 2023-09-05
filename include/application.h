#pragma once

namespace MandelbrotSet {

class Application {
 public:
  virtual ~Application() = default;
  virtual int Run() = 0;
};

}  // namespace MandelbrotSet