#include "mandelbrot/application/config.h"

#include "mandelbrot/core/utils.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace mandelbrot {

// TODO: improve json parsing, remove code duplication
static void ParseValue(const rapidjson::Value& value, double_t& out) {
  out = value.GetDouble();
}

static void ParseValue(const rapidjson::Value& value, bool& out) {
  out = value.GetBool();
}

static void ParseValue(const rapidjson::Value& value, uint32_t& out) {
  out = value.GetUint();
}

static void ParseValue(const rapidjson::Value& value, std::string& out) {
  out = value.GetString();
}

static void ParseValue(const rapidjson::Value& value, Complex& out) {
  ParseValue(value["real"], out.real);
  ParseValue(value["imag"], out.imag);
}

static void ParseValue(const rapidjson::Value& value, Size& out) {
  ParseValue(value["width"], out.width);
  ParseValue(value["height"], out.height);
}

static void ParseValue(const rapidjson::Value& value, WindowMode& out) {
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

static void ParseValue(const rapidjson::Value& value, ColoringMode& out) {
  auto mode = uint32_t{};
  ParseValue(value, mode);
  if (mode >= static_cast<uint32_t>(ColoringMode::kCount)) {
    throw std::runtime_error{"Unexpected ColoringMode"};
  }
  out = static_cast<ColoringMode>(mode);
}

static void ParseValue(const rapidjson::Value& value, Palette& out) {
  auto palette = uint32_t{};
  ParseValue(value, palette);
  if (palette >= static_cast<uint32_t>(Palette::kCount)) {
    throw std::runtime_error{"Unexpected Palette"};
  }
  out = static_cast<Palette>(palette);
}

static auto FindConfigFile() {
  return FindDataFile("config.json");
}

static Config ParseConfig() {
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
  ParseValue(document["palette"], config.palette);
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