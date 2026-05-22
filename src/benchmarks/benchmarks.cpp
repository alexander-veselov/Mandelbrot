#include <benchmark/benchmark.h>

#include "mandelbrot/core/coloring_mode.h"
#include "mandelbrot/core/complex.h"
#include "mandelbrot/core/cuda/mandelbrot_set.h"
#include "mandelbrot/core/image.h"
#include "mandelbrot/core/palette.h"
#include <mandelbrot/core/utils.h>

#include <charconv>

namespace mandelbrot {

static void Default(benchmark::State& state) {
  constexpr auto kSize = Size{1024, 768};
  const auto kCenter = Complex{-0.5, 0.0};
  constexpr auto kZoom = 1.0;
  constexpr auto kMaxIterations = 1024;
  constexpr auto kColoringMode = static_cast<uint32_t>(ColoringMode::kMode1);
  constexpr auto kPalette = static_cast<uint32_t>(Palette::kBluePalette);
  constexpr auto kSmoothing = true;

  auto image = Image{kSize};
  for (auto _ : state) {
    cuda::VisualizeNaive(image.GetData(), image.GetWidth(), image.GetHeight(),
                         kCenter.real, kCenter.imag, kZoom, kMaxIterations,
                         kColoringMode, kPalette, kSmoothing);
  }
}
BENCHMARK(Default)
  ->MinTime(1)
  ->Unit(benchmark::kMillisecond);

static void DifferentResolution(benchmark::State& state) {
  const auto width = static_cast<uint32_t>(state.range(0));
  const auto height = static_cast<uint32_t>(state.range(1));
  const auto size = Size{width, height};
  const auto kCenter = Complex{ -0.5, 0.0 };
  constexpr auto kZoom = 1.0;
  constexpr auto kMaxIterations = 1024;
  constexpr auto kColoringMode = static_cast<uint32_t>(ColoringMode::kMode1);
  constexpr auto kPalette = static_cast<uint32_t>(Palette::kBluePalette);
  constexpr auto kSmoothing = true;

  auto image = Image{size};
  for (auto _ : state) {
    cuda::VisualizeNaive(image.GetData(), image.GetWidth(), image.GetHeight(),
                         kCenter.real, kCenter.imag, kZoom, kMaxIterations,
                         kColoringMode, kPalette, kSmoothing);
  }
}
BENCHMARK(DifferentResolution)
  ->Args({ 640,  360  }) // nHD
  ->Args({ 854,  480  }) // FWVGA
  ->Args({ 960,  540  }) // qHD
  ->Args({ 1024, 576  }) // WSVGA
  ->Args({ 1280, 720  }) // HD
  ->Args({ 1366, 768  }) // FWXGA
  ->Args({ 1600, 900  }) // HD+
  ->Args({ 1920, 1080 }) // Full HD
  ->Args({ 2560, 1440 }) // QHD
  ->Args({ 3200, 1800 }) // QHD+
  ->Args({ 3840, 2160 }) // 4K UHD
  ->Args({ 5120, 2880 }) // 5K
  ->Args({ 7680, 4320 }) // 8K UHD
  ->MinTime(1)
  ->Unit(benchmark::kMillisecond);

static void DifferentMaxIterations(benchmark::State& state) {
  constexpr auto kSize = Size{1024, 768};
  const auto kCenter = Complex{ -0.5, 0.0 };
  constexpr auto kZoom = 1.0;
  const auto max_iterations = static_cast<int32_t>(state.range(0));
  constexpr auto kColoringMode = static_cast<uint32_t>(ColoringMode::kMode1);
  constexpr auto kPalette = static_cast<uint32_t>(Palette::kBluePalette);
  constexpr auto kSmoothing = true;

  auto image = Image{kSize};
  for (auto _ : state) {
    cuda::VisualizeNaive(image.GetData(), image.GetWidth(), image.GetHeight(),
                         kCenter.real, kCenter.imag, kZoom, max_iterations,
                         kColoringMode, kPalette, kSmoothing);
  }
}
BENCHMARK(DifferentMaxIterations)
  ->Arg(128)
  ->Arg(256)
  ->Arg(512)
  ->Arg(1024)
  ->Arg(2048)
  ->Arg(4096)
  ->Arg(8192)
  ->MinTime(1)
  ->Unit(benchmark::kMillisecond);

template <typename Float>
static Float InitFloat(std::string_view literal) {
  if constexpr (std::is_same_v<Float, double_t>) {
    auto v = double_t{};
    std::from_chars(literal.data(), literal.data() + literal.size(), v);
    return v;
  } else {
    return Float(std::string(literal));
  }
}

static void HighZoomNormal(benchmark::State& state) {
  constexpr auto kSize = Size{ 1024, 768 };
  constexpr auto kColoringMode = static_cast<uint32_t>(ColoringMode::kMode1);
  constexpr auto kPalette = static_cast<uint32_t>(Palette::kBluePalette);
  constexpr auto kSmoothing = true;

  const auto real = InitFloat<Float>("-0.2294379204917521921283938659935972297779995325625177334975946836024825031811734391974523087417743062020561864658849527129599780000");
  const auto imag = InitFloat<Float>("+0.8209092363709344023306248042908969588444321104496672178513855603915842881211531455988556895799006496899604886990696093819533990000");
  const auto zoom = InitFloat<Float>("8617594940382931955969294001603998395666896884150973047658560539684745901643990851155924198047262999813551579129376683.12893800950000");
  const auto max_iterations = 4096;

  auto image = Image{ kSize };

  for (auto _ : state) {
    cuda::VisualizePerturbation(image.GetData(), image.GetWidth(), image.GetHeight(),
      real, imag, zoom, max_iterations,
      kColoringMode, kPalette, kSmoothing);
  }
}

BENCHMARK(HighZoomNormal)
  ->Iterations(5)
  ->Unit(benchmark::kMillisecond);

static void HighZoomHeavy(benchmark::State& state) {
  constexpr auto kSize = Size{ 1024, 768 };
  constexpr auto kColoringMode = static_cast<uint32_t>(ColoringMode::kMode1);
  constexpr auto kPalette = static_cast<uint32_t>(Palette::kBluePalette);
  constexpr auto kSmoothing = true;

  const auto real = InitFloat<Float>("-0.7499962814923143790642160359331255832889316887107532993957730552496709780558365928185421048044863443589735578622619681905794452100");
  const auto imag = InitFloat<Float>("-0.0065511461630430211865759781060440814464714161394296919967918757588058752769649158251142223443778126729329264578443902386756515482");
  const auto zoom = InitFloat<Float>("494674475186802996536021892450697064199.575524610587356084897548405961134504813639661199849287243076131546113144383715927074969970000");
  const auto max_iterations = 8192;

  auto image = Image{ kSize };

  for (auto _ : state) {
    cuda::VisualizePerturbation(image.GetData(), image.GetWidth(), image.GetHeight(),
      real, imag, zoom, max_iterations,
      kColoringMode, kPalette, kSmoothing);
  }
}

BENCHMARK(HighZoomHeavy)
  ->Iterations(5)
  ->Unit(benchmark::kMillisecond);

}