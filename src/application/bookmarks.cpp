#include "mandelbrot/application/bookmarks.h"

#include "mandelbrot/core/utils.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace mandelbrot {

Bookmarks::Bookmarks() :
  bookmarks_{Load()},
  current_bookmark_index_{0} {}

Bookmarks::~Bookmarks() {
  // TODO: enable when implemented
  // Save(bookmarks_);
}

void Bookmarks::Add(const Complex& position, double_t zoom) {
  bookmarks_.push_back(Bookmark{position, zoom});
  current_bookmark_index_ = bookmarks_.size() - 1;
}

const Bookmarks::Bookmark& Bookmarks::Next() {
  if (IsEmpty()) {
    throw std::runtime_error{"Bookmarks are empty"};
  }
  if (current_bookmark_index_ == bookmarks_.size() - 1) {
    current_bookmark_index_ = 0;
  } else {
    ++current_bookmark_index_;
  }
  return bookmarks_[current_bookmark_index_];
}

const Bookmarks::Bookmark& Bookmarks::Previous() {
  if (IsEmpty()) {
    throw std::runtime_error{"Bookmarks are empty"};
  }
  if (current_bookmark_index_ == 0) {
    current_bookmark_index_ = bookmarks_.size() - 1;
  } else {
    --current_bookmark_index_;
  }
  return bookmarks_[current_bookmark_index_];
}

const Bookmarks::Bookmark& Bookmarks::Current() const {
  return bookmarks_[current_bookmark_index_];
}

bool Bookmarks::IsEmpty() const {
  return bookmarks_.empty();
}

static auto FindBookmarksFile() {
  return FindDataFile("bookmarks.json");
}

// TODO: improve json parsing, remove code duplication
static void ParseValue(const rapidjson::Value& value, double_t& out) {
  out = value.GetDouble();
}

static void ParseValue(const rapidjson::Value& value, Complex& out) {
  ParseValue(value["real"], out.real);
  ParseValue(value["imag"], out.imag);
}

static void ParseBookmark(const rapidjson::Value& value,
                          Bookmarks::Bookmark& out) {
  ParseValue(value["position"], out.position);
  ParseValue(value["zoom"], out.zoom);
}

static std::vector<Bookmarks::Bookmark> ParseBookmarks(
    const std::filesystem::path& bookmarks_file_path) {

  auto istream = std::ifstream{bookmarks_file_path};
  auto istream_wrapper = rapidjson::IStreamWrapper(istream);

  auto document = rapidjson::Document{};
  document.ParseStream(istream_wrapper);

  auto bookmarks = std::vector<Bookmarks::Bookmark>{};
  for (const auto& bookmark : document["bookmarks"].GetArray()) {
    ParseBookmark(bookmark, bookmarks.emplace_back());
  }

  return bookmarks;
}

static std::vector<Bookmarks::Bookmark> DefaultBookmarks() {
  return std::vector<Bookmarks::Bookmark>{
      Bookmarks::Bookmark{Complex{-0.5, 0.}, 1.0}};
}

std::vector<Bookmarks::Bookmark> Bookmarks::Load() {
  const auto bookmarks_file_path = FindBookmarksFile();
  if (!bookmarks_file_path.has_value()) {
    return DefaultBookmarks();
  }

  return ParseBookmarks(bookmarks_file_path.value());
}

void Bookmarks::Save(const std::vector<Bookmark>& bookmarks) {
  const auto bookmarks_file_path = FindBookmarksFile();
  if (!bookmarks_file_path.has_value()) {
    return;
  }

  // TODO: implement
  throw std::runtime_error{"Not implemented"};
}

}  // namespace mandelbrot