#pragma once

#include <string>

namespace MandelbrotSet {

class Logger {
 public:
   Logger(const Logger&) = delete;
   Logger& operator= (const Logger&) = delete;
   static const Logger& Instance();

   void ResetCursor() const;
   std::string NewLine() const;
   const Logger& ShowSign(bool show) const;
   const Logger& SetPrecision(int64_t precision) const;

   const Logger& operator<<(const Logger&) const;
   const Logger& operator<<(const std::string&) const;
   const Logger& operator<<(const char*) const;
   const Logger& operator<<(int64_t) const;
   const Logger& operator<<(int32_t) const;
   const Logger& operator<<(int16_t) const;
   const Logger& operator<<(int8_t) const;
   const Logger& operator<<(uint64_t) const;
   const Logger& operator<<(uint32_t) const;
   const Logger& operator<<(uint16_t) const;
   const Logger& operator<<(uint8_t) const;
   const Logger& operator<<(double_t) const;
   const Logger& operator<<(float_t) const;
 private:
   Logger();
};

}