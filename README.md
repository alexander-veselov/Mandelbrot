<h3 align="center"><img src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/logo.png" alt="logo" height="200px"></h3>

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

# Used 3rd-party

- GLFW
- CUDA
- RapidJSON
- LodePNG
- Google Test
- Google benchmark

# Images
<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/m0.106750445_m0.88282584799999997_200000.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/0.28893504399999997_0.012374825000000001_4000.png"/>
</p>

<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/0.28601675599999998_m0.011559813_500.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/m1.396216643_0.0042737729999999998_1000.png"/>
</p> 
    
<p align="center">
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/m1.3944205750000001_0.0018272119999999999_5000.png"/>
&nbsp;
    <img width="49%" src="https://github.com/alexander-veselov/Mandelbrot/blob/master/images/0.23533695499999999_m0.51526230500000003_1500.png"/>
</p>
<p>Application supports different coloring modes. You can find more images <a href="https://github.com/alexander-veselov/Mandelbrot/tree/master/images">here</a>.</p> 

# Benchmarks

```
--------------------------------------
Benchmark            Time          FPS
--------------------------------------
640 x 360         2.48 ms          560
960 x 540         4.82 ms          299
1280 x 720        8.77 ms          176
1366 x 768        8.92 ms          157
1920 x 1080       16.6 ms           81
2560 x 1440       28.5 ms           47
3840 x 2160       62.6 ms           22
7680 x 4320        243 ms            6
```

# Controls
Navigation:
- drag and move mouse to navigate
- mouse scroll up to zoom in
- mouse scroll down to zoom out

Bookmarks:
- ←: previous bookmark
- →: next bookmark
- ↑: navigate to current bookmark
- ↓: create a bookmark

Coloring:
- m: change coloring mode
- p: change palette

Iterations:
- comma: decrease iterations count
- period: increase iterations count

Application:
- escape: close application
