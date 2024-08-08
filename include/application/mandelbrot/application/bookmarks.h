#pragma once

#include "mandelbrot/core/complex.h"

#include <vector>

namespace mandelbrot {

class Bookmarks {
 public:
  struct Bookmark {
    Complex position;
    double_t zoom;
  };

  Bookmarks();
  ~Bookmarks();
  void Add(const Complex& position, double_t zoom);
  const Bookmark& Next();
  const Bookmark& Previous();
  const Bookmark& Current() const;
  bool IsEmpty() const;

 private:
  std::vector<Bookmark> Load();
  void Save(const std::vector<Bookmark>& bookmarks);

 private:
  std::vector<Bookmark> bookmarks_;
  size_t current_bookmark_index_;
};

}