<h3 align="center"><img src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/logo.png" alt="logo" height="200px"></h3>

<div align="center">
  
# Mandelbrot

</div>

<p align="center">CUDA-accelerated Mandelbrot set explorer</p>
<h4 align="center">

  
  <a href="https://github.com/alexander-veselov/Mandelbrot/commits/master/">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/alexander-veselov/Mandelbrot">
  </a>
  <a href="https://github.com/alexander-veselov/Mandelbrot/blob/master/.github/workflows/cmake-windows.yml">
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/alexander-veselov/Mandelbrot/cmake-windows.yml">
  </a>
</h4>

# Dependencies

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [GLFW](https://github.com/glfw/glfw)
- [RapidJSON](https://github.com/Tencent/rapidjson)
- [LodePNG](https://github.com/lvandeve/lodepng)
- [Google Test](https://github.com/google/googletest)
- [Google Benchmark](https://github.com/google/benchmark)

# Gallery
<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/m0.106750445_m0.88282584799999997_200000.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/0.28893504399999997_0.012374825000000001_4000.png"/>
</p>

<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/0.28601675599999998_m0.011559813_500.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/m1.396216643_0.0042737729999999998_1000.png"/>
</p> 
    
<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/m1.3944205750000001_0.0018272119999999999_5000.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/data/images/0.23533695499999999_m0.51526230500000003_1500.png"/>
</p>
<p>Application supports different coloring modes. You can find more images <a href="https://github.com/alexander-veselov/Mandelbrot/tree/master/images">here</a>.</p> 

# Benchmarks
```
GPU: GeForce RTX 4070 Ti

----------------------------------------       ----------------------------------------
Double Precision    Time             FPS       Single Precision    Time             FPS
----------------------------------------       ----------------------------------------
640 x 360           2.36 ms          597       640 x 360          0.253 ms         5600
960 x 540           4.67 ms          299       960 x 540          0.440 ms         3200
1280 x 720          7.86 ms          179       1280 x 720         0.706 ms         1948
1366 x 768          8.70 ms          160       1366 x 768         0.800 ms         1757
1920 x 1080         16.3 ms           81       1920 x 1080         1.57 ms          815
2560 x 1440         27.8 ms           50       2560 x 1440         2.78 ms          527
3840 x 2160         61.2 ms           23       3840 x 2160         5.80 ms          242
7680 x 4320          239 ms            6       7680 x 4320         23.6 ms           64
```
