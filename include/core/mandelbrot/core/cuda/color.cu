__device__ constexpr uint32_t MakeRGB(uint8_t r, uint8_t g, uint8_t b,
                                      uint8_t a) {
  return r + (g << 8) + (b << 16) + (a << 24);
}

__device__ constexpr uint32_t InterpolateColor(uint32_t color1, uint32_t color2,
                                               double_t fraction) {
  const auto b1 = static_cast<uint8_t>((color1 >> 16) & 0xff);
  const auto b2 = static_cast<uint8_t>((color2 >> 16) & 0xff);
  const auto g1 = static_cast<uint8_t>((color1 >> 8) & 0xff);
  const auto g2 = static_cast<uint8_t>((color2 >> 8) & 0xff);
  const auto r1 = static_cast<uint8_t>(color1 & 0xff);
  const auto r2 = static_cast<uint8_t>(color2 & 0xff);

  const auto r = static_cast<uint32_t>((r2 - r1) * fraction + r1);
  const auto g = static_cast<uint32_t>((g2 - g1) * fraction + g1) << 8;
  const auto b = static_cast<uint32_t>((b2 - b1) * fraction + b1) << 16;
  constexpr auto kAlpha = static_cast<uint32_t>(uint8_t{255} << 24);

  return r + g + b + kAlpha;
}

__device__ uint32_t LchToRGB(double L, double C, double H) {
  // Convert Lch to Lab
  auto A = C * cos(H * CUDART_PI_F / 180.0);
  auto B = C * sin(H * CUDART_PI_F / 180.0);

  // Convert Lab to XYZ
  auto Y = (L + 16.) / 116.;
  auto X = A / 500. + Y;
  auto Z = Y - B / 200.;

  if (pow(Y, 3) > 0.008856) {
    Y = pow(Y, 3);
  } else {
    Y = (Y - 16. / 116.) / 7.787;
  }

  if (pow(X, 3) > 0.008856) {
    X = pow(X, 3);
  } else {
    X = (X - 16. / 116.) / 7.787;
  }

  if (pow(Z, 3) > 0.008856) {
    Z = pow(Z, 3);
  } else {
    Z = (Z - 16. / 116.) / 7.787;
  }

  X *= 0.95047;
  Y *= 1.00000;
  Z *= 1.08883;

  // Convert XYZ to RGB
  auto R = X * +3.2406 + Y * -1.5372 + Z * -0.4986;
  auto G = X * -0.9689 + Y * +1.8758 + Z * +0.0415;
  /**/ B = X * +0.0557 + Y * -0.2040 + Z * +1.0570;

  if (R > 0.0031308) {
    R = 1.055 * pow(R, (1 / 2.4)) - 0.055;
  } else {
    R = 12.92 * R;
  }

  if (G > 0.0031308) {
    G = 1.055 * pow(G, (1 / 2.4)) - 0.055;
  } else {
    G = 12.92 * G;
  }

  if (B > 0.0031308) {
    B = 1.055 * pow(B, (1 / 2.4)) - 0.055;
  } else {
    B = 12.92 * B;
  }

  return MakeRGB(R * UINT8_MAX, G * UINT8_MAX, B * UINT8_MAX);
}

__device__ uint32_t HSVToRGB(double h, double s, double v) {
  if (s <= 0.0) {
    return MakeRGB(v * UINT8_MAX, v * UINT8_MAX, v * UINT8_MAX);
  }

  auto hh = h;
  if (hh >= 360.0) {
    hh = 0.0;
  }
  hh /= 60.0;

  const auto i = static_cast<uint8_t>(hh);
  const auto ff = hh - i;
  const auto p = v * (1.0 - s);
  const auto q = v * (1.0 - (s * ff));
  const auto t = v * (1.0 - (s * (1.0 - ff)));

  auto R = double_t{};
  auto G = double_t{};
  auto B = double_t{};

  switch (i) {
    case 0:
      R = v;
      G = t;
      B = p;
      break;
    case 1:
      R = q;
      G = v;
      B = p;
      break;
    case 2:
      R = p;
      G = v;
      B = t;
      break;
    case 3:
      R = p;
      G = q;
      B = v;
      break;
    case 4:
      R = t;
      G = p;
      B = v;
      break;
    case 5:
    default:
      R = v;
      G = p;
      B = q;
      break;
  }

  return MakeRGB(R * UINT8_MAX, G * UINT8_MAX, B * UINT8_MAX);
}