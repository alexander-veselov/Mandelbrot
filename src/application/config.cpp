#include "mandelbrot/application/config.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace MandelbrotSet {

std::optional<std::filesystem::path> FindConfigFile() {
  constexpr auto kConfigFilename = "config.json";
  const auto directory_candidate1 = std::filesystem::current_path();
  const auto directory_candidate2 = std::filesystem::current_path() / "data";
  const auto candidate1 = directory_candidate1 / kConfigFilename;
  if (std::filesystem::exists(candidate1)) {
    return candidate1;
  }
  const auto candidate2 = directory_candidate2 / kConfigFilename;
  if (std::filesystem::exists(candidate2)) {
    return candidate2;
  }
  return std::nullopt;
}

void ParseValue(const rapidjson::Value& value, double_t& out) {
  out = value.GetDouble();
}

void ParseValue(const rapidjson::Value& value, bool& out) {
  out = value.GetBool();
}

void ParseValue(const rapidjson::Value& value, uint32_t& out) {
  out = value.GetUint();
}

void ParseValue(const rapidjson::Value& value, std::string& out) {
  out = value.GetString();
}

void ParseValue(const rapidjson::Value& value, Complex& out) {
  ParseValue(value["real"], out.real);
  ParseValue(value["imag"], out.imag);
}

void ParseValue(const rapidjson::Value& value, Size& out) {
  ParseValue(value["width"], out.width);
  ParseValue(value["height"], out.height);
}

void ParseValue(const rapidjson::Value& value, WindowMode& out) {
  auto window_mode = std::string{};
  ParseValue(value, window_mode);
  if (window_mode == "windowed") {
    out = WindowMode::kWindowed;
  } else if (window_mode == "fullscreen") {
    out = WindowMode::kFullscreen;
  } else if (window_mode == "borderless") {
    out = WindowMode::kBorderless;
  } else {
    throw std::runtime_error{"Unexpected WindowMode"};
  }
}

void ParseValue(const rapidjson::Value& value, ColoringMode& out) {
  auto window_mode = std::string{};
  ParseValue(value, window_mode);
  if (window_mode == "blackwhite") {
    out = ColoringMode::kBlackWhite;
  } else if (window_mode == "blue") {
    out = ColoringMode::kBlue;
  } else if (window_mode == "red") {
    out = ColoringMode::kRed;
  } else if (window_mode == "bluegreen") {
    out = ColoringMode::kBlueGreen;
  } else if (window_mode == "orange") {
    out = ColoringMode::kOrange;
  } else if (window_mode == "waves") {
    out = ColoringMode::kWaves;
  } else {
    throw std::runtime_error{"Unexpected ColoringMode"};
  }
}

Config ParseConfig() {
  const auto config_file_path = FindConfigFile();
  if (!config_file_path.has_value()) {
    return Config{};
  }

  auto istream = std::ifstream{config_file_path.value()};
  auto istream_wrapper = rapidjson::IStreamWrapper(istream);

  auto document = rapidjson::Document{};
  document.ParseStream(istream_wrapper);

  auto config = Config{};
  ParseValue(document["coloring_mode"], config.coloring_mode);
  ParseValue(document["default_position"], config.default_position);
  ParseValue(document["default_zoom"], config.default_zoom);
  ParseValue(document["directional_zoom"], config.directional_zoom);
  ParseValue(document["enable_vsync"], config.enable_vsync);
  ParseValue(document["fps_update_rate"], config.fps_update_rate);
  ParseValue(document["max_iterations"], config.max_iterations);
  ParseValue(document["screenshot_size"], config.screenshot_size);
  ParseValue(document["screenshots_folder"], config.screenshots_folder);
  ParseValue(document["smoothing"], config.smoothing);
  ParseValue(document["window_mode"], config.window_mode);
  ParseValue(document["window_size"], config.window_size);
  ParseValue(document["zoom_factor"], config.zoom_factor);

  return config;
}

const Config& GetConfig() {
  static auto config = ParseConfig();
  return config;
}

}