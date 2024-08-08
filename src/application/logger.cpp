#include "mandelbrot/application/logger.h"

#include <iostream>
#include <iomanip>

#ifdef WIN32
#include <windows.h>
#endif

namespace MandelbrotSet {

void HideCursor() {
#ifdef WIN32
  const auto console_handle = GetStdHandle(STD_OUTPUT_HANDLE);
  auto info = CONSOLE_CURSOR_INFO{};
  info.dwSize = 100;
  info.bVisible = FALSE;
  SetConsoleCursorInfo(console_handle, &info);
#else
  throw std::runtime_error{ "HideCursor not implemented on this platform" };
#endif
}

void GotoXY(int32_t x, int32_t y) {
#ifdef WIN32
  const auto console = GetStdHandle(STD_OUTPUT_HANDLE);
  auto coord = COORD{};
  coord.X = x;
  coord.Y = y;
  SetConsoleCursorPosition(console, coord);
#else
  throw std::runtime_error{ "GotoXY not implemented on this platform" };
#endif
}

Logger::Logger() {
  HideCursor();
}

void Logger::ResetCursor() const {
  GotoXY(0, 0);
}

std::string Logger::NewLine() const {
  auto newline = std::string{};
  constexpr auto kSpacesCount = 30;
  for (auto i = 0; i < kSpacesCount; ++i) {
    newline += ' ';
  }
  return newline + '\n';
}

const Logger& Logger::ShowSign(bool show) const {
  if (show) {
    std::cout << std::showpos;
  }
  else {
    std::cout << std::noshowpos;
  }
  return *this;
}

const Logger& Logger::SetPrecision(int64_t precision) const {
  std::cout << std::setprecision(precision);
  return *this;
}

const Logger& Logger::operator<<(const Logger& logger) const {
  return logger;
}

const Logger& Logger::operator<<(const std::string& data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(const char* data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(int64_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(int32_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(int16_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(int8_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(uint64_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(uint32_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(uint16_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(uint8_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(double_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::operator<<(float_t data) const {
  std::cout << data;
  return *this;
}

const Logger& Logger::Instance() {
  static auto instance = Logger{};
  return instance;
}

}