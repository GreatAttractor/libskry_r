# **libskry_r**

## Lucky imaging library

Copyright (C) 2017 Filip Szczerek (ga.software@yahoo.com)

*This project is licensed under the terms of the MIT license (see the LICENSE file for details).*

----------------------------------------

- 1\. Introduction
- 2\. Performance comparison with C
  - 2\.1\. Implementation remarks
- 3\. Input/output formats support
- 4\. Principles of operation
- 5\. Change log


----------------------------------------
## 1. Introduction

**libskry_r** implements the *lucky imaging* principle of astronomical imaging: creating a high-quality still image out of a series of many (possibly thousands) low quality ones (blurred, deformed, noisy). The resulting *image stack* typically requires post-processing, including sharpening (e.g. via deconvolution in [***ImPPG***](http://greatattractor.github.io/imppg/)).

*libskry_r* is a Rust rewrite of [***libskry***](https://github.com/GreatAttractor/libskry), mostly complete. Not yet ported: multi-threading, demosaicing, and video file support via *libav*.

For a visualization of the stacking process, see the [**Stackistry video tutorial**](https://www.youtube.com/watch?v=_68kEYBXkLw&list=PLCKkDZ7up_-VRMzGQ0bmmiXL39z78zwdE).

For sample results, see the [**gallery**](https://www.astrobin.com/users/GreatAttractor/collections/131/).

See `doc/example1.rs` for a usage example.

See also the [**Algorithms summary**](https://github.com/GreatAttractor/libskry/raw/master/doc/algorithms.pdf) in *libskry* repository.


----------------------------------------
## 2. Performance comparison with C

The two goals of rewriting *libskry* were practising Rust and comparing the performance to C99 code.

The following figures were obtained on a Core i5-3570K under Fedora 25. Each test was preceded by a pre-run to make sure the input video is cached in RAM. In case of *libskry*, the program was `doc/example1.c`. In case of *libskry_r*, it was `doc/example1.rs`. Both use the same processing parameters. The raw video `sun01.avi` (840x612 8 bpp mono, 634 frames, Sun in Hα) can be downloaded in the “Releases” section. The C program was forced to use 1 thread with `OMP_NUM_THREADS=1`.


| Compiler             | Options                                                                     | Execution time |
|----------------------|-----------------------------------------------------------------------------|----------------|
| rustc 1.23.0-nightly | `RUSTFLAGS="-C opt-level=3 -C target-cpu=native"`, `cargo` with `--release` | 23.4 s         |
| GCC 6.4.1            | `-O3 -ffast-math -march=native`                                             | 25.7 s         |
| Clang 3.9.1          | `-O3 -ffast-math -march=native`                                             | 23.9 s         |


### 2.1 Implementation remarks

The processing is not very complex. The calculations include: iterated box blur, block matching (both on 8-bit mono images), bilinear interpolation (on 32-bit floating-point pixel values) and Delaunay triangulation (only once, typically for a few hundred to a few thousand points, with negligible time impact). The only collection type used is vector.

The most time-consuming operation is block matching. Replacing the inner loop’s body in `blk_match::calc_sum_of_squared_diffs`:

```Rust
result += sqr!(img_pix[img_offs + x as usize] as i32 -
               rblk_pix[blk_offs + x as usize] as i32) as u64;
```

with an unsafe block:


```Rust
unsafe {
    result += sqr!(*img_pix.get_unchecked(img_offs + x as usize) as i32 -
                   *rblk_pix.get_unchecked(blk_offs + x as usize) as i32) as u64;
}
```

enabled the compiler to vectorize it (using AVX) and reduced the reference point alignment phase’s execution time of `example1.rs` from 26.7 s to 12.6 s.

On the other hand, doing the same in `filters::box_blur_pass` and `filters::estimate_quality` had negligible effect on the quality estimation speed.

Other uses of `unsafe` are not performance-critical: pixel format conversions, reading structures from a file, creating an uninitialized vector.

In *libskry*, the user is asked to create the processing phase objects in appropriate order and not to modify the underlying image sequence while processing is in progress. In *libskry_r*, correct usage in enforced by Rust’s borrow checker; it also necessitated modification of the API. Each processing phase is now represented by a `__Proc` struct, which holds a mutable reference to the input image sequence, and a `__Data` struct with this phase’s results (which does not hold any references). Each `__Proc` is created in a sub-scope; each resulting `__Data` is then fed to the next phase’s `__Proc`. See `doc/example1.rs` for details.


----------------------------------------
## 3. Input/output formats support

Supported input formats:

- AVI: uncompressed DIB (mono or RGB), Y8/Y800
- SER: mono, RGB
- BMP: 8-, 24- and 32-bit uncompressed
- TIFF: 8- and 16-bit per channel mono or RGB uncompressed

Supported output formats:

- BMP: 8- and 24-bit uncompressed
- TIFF: 8- and 16-bit per channel mono or RGB uncompressed

At the moment there is only limited AVI support (no extended or ODML AVI headers).


----------------------------------------
## 4. Principles of operation

Processing of a raw input image sequence consists of the following steps:

1. Image alignment (video stabilization)
2. Quality estimation
3. Reference point alignment
4. Image stacking

**Image alignment** compensates any global image drift; the result is a stabilized video of size usually smaller than any of the input images. The (rectangular) region visible in all input images is referred to as *images’ intersection* throughout the source code.

**Quality estimation** concerns the changes of local image quality. This information is later used to reject (via an user-specified criterion) poor-quality image fragments during reference point alignment and image stacking.

**Reference point alignment** traces the geometric distortion of images (by using local block matching) which is later compensated for during image stacking.

**Image stacking** performs shift-and-add summation of image fragments using information from previous steps. This improves signal-to-noise ratio. Note that stacking too many images may decrease quality – adding lower-quality fragments causes more blurring in the output stack.


----------------------------------------
## 5. Change log

- 0.3.0 (2017-12-05)
  - Initial rewrite (from C) of *libskry* 0.3.0 (a805d0c4a).
