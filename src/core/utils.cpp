#include "utils.h"
#include <fstream>

namespace MandelbrotSet {

#pragma pack(push, 1)  // Sets the alignment to 1 byte
struct BMPFileHeader {
  uint16_t file_type{0x4D42};  // File type, always "BM" which is 0x4D42
  uint32_t file_size{0};       // Size of the file in bytes
  uint16_t reserved1{0};       // Reserved, always 0
  uint16_t reserved2{0};       // Reserved, always 0
  uint32_t offset_data{0};     // Start position of pixel data (54 bytes)
};

struct BMPInfoHeader {
  uint32_t size{40};              // Size of this header (40 bytes)
  int32_t width{0};               // Width of the bitmap in pixels
  int32_t height{0};              // Height of the bitmap in pixels
  uint16_t planes{1};             // Number of color planes, must be 1
  uint16_t bit_count{24};         // Bits per pixel (24 for RGB)
  uint32_t compression{0};        // Compression type (0 for no compression)
  uint32_t size_image{0};         // Size of the image data
  int32_t x_pixels_per_meter{0};  // Horizontal resolution in pixels per meter
  int32_t y_pixels_per_meter{0};  // Vertical resolution in pixels per meter
  uint32_t colors_used{0};        // Number of colors in the color palette
  uint32_t colors_important{0};   // Important colors (0 = all)
};
#pragma pack(pop)

void WriteImage(const Image& image, const std::filesystem::path& path) {

  const auto width = image.GetWidth();
  const auto height = image.GetHeight();

  if (width == 0 || height == 0) {
    throw std::runtime_error{"Wrong image size"};
  }

  constexpr auto kBMPExtension = ".bmp";
  if (path.filename().extension() != kBMPExtension) {
    throw std::invalid_argument("Unsupported image format");
  }

  if (!std::filesystem::exists(path.parent_path())) {
    std::filesystem::create_directory(path.parent_path());
  }

  auto file = std::ofstream{path, std::ios::out | std::ios::binary};
  if (!file) {
    throw std::runtime_error{"Cannot open file: " + path.string()};
  }

  const auto row_size = (width * 3 + 3) & (~3);  // Each row is aligned to a multiple of 4 bytes

  auto info_header = BMPInfoHeader{};
  info_header.width = width;
  info_header.height = height;
  info_header.size_image = row_size * height;

  auto file_header = BMPFileHeader{};
  file_header.file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + info_header.size_image;
  file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

  file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
  file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));

  const auto data = image.GetData();
  for (auto y = 0; y < height; ++y) {
    for (auto x = 0; x < width; ++x) {
      const auto rgba = data[y * width + x];
      const auto b = static_cast<uint8_t>((rgba >> 16) & 0xff);
      const auto g = static_cast<uint8_t>((rgba >> 8) & 0xff);
      const auto r = static_cast<uint8_t>(rgba & 0xff);

      file.write(reinterpret_cast<const char*>(&b), sizeof(char));
      file.write(reinterpret_cast<const char*>(&g), sizeof(char));
      file.write(reinterpret_cast<const char*>(&r), sizeof(char));
    }
    file.write("\0\0\0", row_size - width * 3);  // Padding for each row
  }

  file.close();
}

}  // namespace MandelbrotSet