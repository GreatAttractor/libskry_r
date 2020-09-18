//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Processing phase: image stacking (shift-and-add summation).
//

use defs::{Point, PointFlt, ProcessingPhase, ProcessingError, Rect};
use image;
use image::{DemosaicMethod, Image, ImageError, PixelFormat};
use img_align::ImgAlignmentData;
use img_seq::{ImageSequence, SeekResult};
use ref_pt_align::RefPointAlignmentData;
use std::cmp::{min, max};


struct StackTrianglePoint {
    // Image coordinates in the stack
    pub x: i32,
    pub y: i32,

    // Barycentric coordinates in the parent triangle
    pub u: f32,
    pub v: f32
}


/// Performs image stacking (shift-and-add summation).
pub struct StackingProc<'a> {
    img_seq: &'a mut ImageSequence,

    img_align_data: &'a ImgAlignmentData,

    ref_pt_align_data: &'a RefPointAlignmentData,

    is_complete: bool,

    /// For each triangle in `ref_pt_align_data.triangulation`, contains a list of points comprising it.
    rasterized_tris: Vec<Vec<StackTrianglePoint>>,

    /// Final positions (within the images' intersection) of the reference points,
    /// i.e. the average over all images where the points are valid.
    final_ref_pt_pos: Vec<PointFlt>,

    /// Element [i] = number of images that were stacked to produce the i-th pixel in `image_stack'.
    added_img_count: Vec<usize>,

    /// Format: Mono32f or RGB32f.
    image_stack: Image,

    first_step_complete: bool,

    /// Triangle indices (from `ref_pt_align_data.triangulation`) stacked in the current step.
    curr_step_stacked_triangles: Vec<usize>,

    /// Contains inverted flat-field values (1/flat-field)
    flatfield_inv: Option<Image>
}


impl<'a> StackingProc<'a> {
    pub fn init(img_seq: &'a mut ImageSequence,
                img_align_data: &'a ImgAlignmentData,
                ref_pt_align_data: &'a RefPointAlignmentData,
                flatfield: Option<Image>)
        -> Result<StackingProc<'a>, ImageError> {

        img_seq.seek_start();

        let final_ref_pt_pos = ref_pt_align_data.get_final_positions();

        let triangulation = &ref_pt_align_data.get_ref_pts_triangulation();
        let mut rasterized_tris = Vec::<Vec<StackTrianglePoint>>::with_capacity(triangulation.get_triangles().len());
        let curr_step_stacked_triangles = Vec::<usize>::with_capacity(triangulation.get_triangles().len());

        let intersection = img_align_data.get_intersection();
        let mut pixel_occupied = vec![false; (intersection.width * intersection.height) as usize];

        for tri in triangulation.get_triangles() {
            rasterized_tris.push(
                rasterize_triangle(
                    &final_ref_pt_pos[tri.v0],
                    &final_ref_pt_pos[tri.v1],
                    &final_ref_pt_pos[tri.v2],
                    Rect{ x: 0, y: 0, width: intersection.width, height: intersection.height },
                    &mut pixel_occupied));
        }
        //TODO: see if after rasterization there are any pixels not belonging to any triangle and assign them

        let (_, _, pix_fmt, _) = img_seq.get_curr_img_metadata()?;

        let stack_pix_fmt = if image::get_num_channels(pix_fmt) == 1 && !pix_fmt.is_cfa() {
                                PixelFormat::Mono32f
                            } else {
                                PixelFormat::RGB32f
                            };

        let image_stack = Image::new(intersection.width, intersection.height, stack_pix_fmt, None, true);

        let added_img_count = vec![0usize; (intersection.width * intersection.height) as usize];

        let mut flatfield_inv: Option<Image> = None;

        if flatfield.is_some() {
            let mut ffield = flatfield.unwrap();

            let ffield_inv =
                if ffield.get_pixel_format() == PixelFormat::Mono32f {
                    ffield.get_copy()
                } else {
                    ffield.convert_pix_fmt(PixelFormat::Mono32f, Some(DemosaicMethod::HqLinear))
                };

            let pixels = ffield.get_pixels_mut::<f32>();
            let max_val = pixels.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap().clone();

            for pix in pixels.iter_mut() {
                if *pix > 0.0 {
                    *pix = max_val / *pix;
                }
            }

            flatfield_inv = Some(ffield_inv);
        }

        Ok(StackingProc{
            img_seq,
            img_align_data,
            ref_pt_align_data,
            is_complete: false,
            rasterized_tris,
            final_ref_pt_pos,
            added_img_count,
            image_stack,
            first_step_complete: false,
            curr_step_stacked_triangles,
            flatfield_inv
        })
    }


    /// Returns the image stack; can be used only after stacking completes.
    pub fn get_image_stack(&self) -> &Image {
        assert!(self.is_complete);
        &self.image_stack
    }


    /// Returns an incomplete image stack, updated after every stacking step.
    pub fn get_partial_image_stack(&self) -> Image {
        let mut pstack = self.image_stack.get_copy();
        normalize_image_stack(&self.added_img_count, &mut pstack, self.flatfield_inv.is_some());
        pstack
    }


    pub fn is_complete(&self) -> bool {
        self.is_complete
    }


    /// Returns an array of triangle indices stacked in current step.
    ///
    /// Meant to be called right after `step()`. Values are indices into triangle array
    /// of the triangulation returned by `RefPointAlignmentData.get_ref_pts_triangulation()`.
    /// Vertex coordinates do not correspond to the triangulation, but to the array
    /// returned by `get_ref_pt_stacking_pos()`.
    ///
    pub fn get_curr_step_stacked_triangles(&self) -> &[usize] {
        &self.curr_step_stacked_triangles
    }

    /// Returns reference point positions as used during stacking.
    pub fn get_ref_pt_stacking_pos(&self) -> &[PointFlt] {
        &self.final_ref_pt_pos
    }
}


impl<'a> ProcessingPhase for StackingProc<'a>
{
    fn get_curr_img(&mut self) -> Result<Image, ImageError> {
        self.img_seq.get_curr_img()
    }


    fn step(&mut self) -> Result<(), ProcessingError> {
        if self.first_step_complete {
            match self.img_seq.seek_next() {
                Err(SeekResult::NoMoreImages) => {
                    normalize_image_stack(&self.added_img_count, &mut self.image_stack, self.flatfield_inv.is_some());
                    self.is_complete = true;
                    return Err(ProcessingError::NoMoreSteps);
                },
                _ => ()
            }
        }

        let mut img = self.img_seq.get_curr_img()?;
        let curr_img_idx = self.img_seq.get_curr_img_idx_within_active_subset();

        let intersection = self.img_align_data.get_intersection();
        let alignment_ofs = self.img_align_data.get_image_ofs()[curr_img_idx];

        if img.get_pixel_format() != self.image_stack.get_pixel_format() {
            img = img.convert_pix_fmt(self.image_stack.get_pixel_format(), Some(DemosaicMethod::HqLinear));
        }

        let num_channels = image::get_num_channels(img.get_pixel_format());

        // For each triangle, check if its vertices are valid in the current image. If they are,
        // add the triangle's contents to the corresponding triangle patch in the stack.

        self.curr_step_stacked_triangles.clear();

        let triangulation = self.ref_pt_align_data.get_ref_pts_triangulation();

        let envelope = Rect{ x: 0, y: 0, width: intersection.width, height: intersection.height };

        // First, find the list of triangles valid in the current step

        for (tri_idx, tri) in triangulation.get_triangles().iter().enumerate() {
            let p0 = self.ref_pt_align_data.get_ref_pt_pos(tri.v0, curr_img_idx);
            let p1 = self.ref_pt_align_data.get_ref_pt_pos(tri.v1, curr_img_idx);
            let p2 = self.ref_pt_align_data.get_ref_pt_pos(tri.v2, curr_img_idx);

            if p0.is_valid && p1.is_valid && p2.is_valid {
                // Due to the way reference point alignment works, it is allowed for a point
                // to be outside the image intersection at some times. Must be careful not to
                // try interpolating pixel values from outside the current image.
                // (Cannot use `intersection` here directly, because its origin may not be (0,0),
                // and `p0`, `p1`, `p2` have coordinates relative to intersection's origin.)
                let p0_inside = envelope.contains_point(&p0.pos);
                let p1_inside = envelope.contains_point(&p1.pos);
                let p2_inside = envelope.contains_point(&p2.pos);

                if p0_inside || p1_inside || p2_inside {
                    self.curr_step_stacked_triangles.push(tri_idx);
                }
            }
        }

        // Second, stack the triangles

        let src_pixels = img.get_pixels::<f32>();

        let stack_stride = self.image_stack.get_width() as usize * num_channels;
        let stack_pixels = self.image_stack.get_pixels_mut::<f32>();

        let mut flatf_pixels: &[f32] = &Vec::<f32>::new()[..];
        let mut flatf_stride = 0;

        if self.flatfield_inv.is_some() {
            let ff_inv: &Image = self.flatfield_inv.iter().next().unwrap();
            flatf_pixels = ff_inv.get_pixels::<f32>();
            flatf_stride = num_channels * ff_inv.get_width() as usize;
        };

//        #pragma omp parallel for
        for tri_idx in &self.curr_step_stacked_triangles {
            let tri = &triangulation.get_triangles()[*tri_idx];

            let p0 = self.ref_pt_align_data.get_ref_pt_pos(tri.v0, curr_img_idx);
            let p1 = self.ref_pt_align_data.get_ref_pt_pos(tri.v1, curr_img_idx);
            let p2 = self.ref_pt_align_data.get_ref_pt_pos(tri.v2, curr_img_idx);

            let p0_inside = envelope.contains_point(&p0.pos);
            let p1_inside = envelope.contains_point(&p1.pos);
            let p2_inside = envelope.contains_point(&p2.pos);
            let all_inside = p0_inside && p1_inside && p2_inside;

            for stp in &self.rasterized_tris[*tri_idx] {
                let srcx = stp.u * p0.pos.x as f32 +
                           stp.v * p1.pos.x as f32 +
                           (1.0 - stp.u - stp.v) * p2.pos.x as f32;
                let srcy = stp.u * p0.pos.y as f32 +
                           stp.v * p1.pos.y as f32 +
                           (1.0 - stp.u - stp.v) * p2.pos.y as f32;

                if all_inside ||
                   srcx >= 0.0 && srcx <= (intersection.width - 1) as f32 &&
                   srcy >= 0.0 && srcy <= (intersection.height - 1) as f32 {

                    let mut ffx = 0;
                    let mut ffy = 0;
                    if self.flatfield_inv.is_some() {
                        ffx = min(srcx as i32 + intersection.x + alignment_ofs.x, self.flatfield_inv.iter().next().unwrap().get_width() as i32 - 1);
                        ffy = min(srcy as i32 + intersection.y + alignment_ofs.y, self.flatfield_inv.iter().next().unwrap().get_height() as i32 - 1);
                    }

                    for ch in 0..num_channels {
                        let mut src_val =
                            interpolate_pixel_value(src_pixels, img.get_width(), img.get_height(),
                                                    srcx + (intersection.x + alignment_ofs.x) as f32,
                                                    srcy + (intersection.y + alignment_ofs.y) as f32,
                                                    ch, num_channels);

                        if self.flatfield_inv.is_some() {
                            // `flatfield_inv` contains inverted flat-field values,
                            // so we multiply instead of dividing
                            src_val *= flatf_pixels[ffy as usize * flatf_stride + ffx as usize * num_channels];
                        }

                        stack_pixels[stp.y as usize * stack_stride + stp.x as usize * num_channels + ch] += src_val;
                    }

                    self.added_img_count[(stp.x + stp.y * intersection.width as i32) as usize] += 1;
                }
            }
        }

        if !self.first_step_complete {
            self.first_step_complete = true;
        }

        Ok(())
    }
}


/// Returns list of pixels belonging to triangle `(v0, v1, v2)`.
///
/// # Parameters
///
/// * `envelope` - Image region corresponding to `pixel_occupied`.
/// * `pixel_occupied` - Pixels of `envelope` (row-major order). If a pixel belongs
///                      to the rasterized triangle, will be set to `true`.
///
fn rasterize_triangle(
    v0: &PointFlt,
    v1: &PointFlt,
    v2: &PointFlt,
    envelope: Rect,
    pixel_occupied: &mut[bool]) -> Vec<StackTrianglePoint> {

    // Test every point of the rectangular axis-aligned bounding box of
    // the triangle (v0, v1, v2) and if it is inside triangle, add it
    // to the returned list.

    let mut points: Vec<StackTrianglePoint> = vec![];

    let xmin = *[v0.x as i32, v1.x as i32, v2.x as i32].iter().min().unwrap();
    let xmax = *[v0.x as i32, v1.x as i32, v2.x as i32].iter().max().unwrap();
    let ymin = *[v0.y as i32, v1.y as i32, v2.y as i32].iter().min().unwrap();
    let ymax = *[v0.y as i32, v1.y as i32, v2.y as i32].iter().max().unwrap();

    for y in ymin .. ymax + 1 {
        for x in xmin .. xmax + 1 {
            if envelope.contains_point(&Point{ x, y }) {
                let is_pix_occupied = &mut pixel_occupied[(x - envelope.x + (y - envelope.y) * envelope.width as i32) as usize];
                if !*is_pix_occupied {
                    let (u, v) = calc_barycentric_coords!(Point{ x, y }, v0, v1, v2);
                    if u >= 0.0 && u <= 1.0 &&
                       v >= 0.0 && v <= 1.0 &&
                       u+v >= 0.0 && u+v <= 1.0 {

                        points.push(StackTrianglePoint{ x, y, u, v });
                        *is_pix_occupied = true;
                    }
                }
            }
        }
    }

    points
}


/// Normalizes `img_stack` using the specified counts of stacked images for each pixel.
fn normalize_image_stack(added_img_count: &[usize], img_stack: &mut Image, uses_flatfield: bool) {
    let mut max_stack_value = 0.0;

    let num_channels = image::get_num_channels(img_stack.get_pixel_format());

    for (i, pix) in img_stack.get_pixels_mut::<f32>().iter_mut().enumerate() {

        *pix /= max(1usize, added_img_count[i / num_channels]) as f32;

        if uses_flatfield && *pix > max_stack_value {
            max_stack_value = *pix;
        }

    }
    if uses_flatfield && max_stack_value > 0.0 {
        for pix in img_stack.get_pixels_mut::<f32>() {
            *pix /= max_stack_value;
        }
    }
}


/// Performs linear interpolation of floating-point pixel values.
pub fn interpolate_pixel_value(
    pixels: &[f32],
    img_width: u32,
    img_height: u32,
    x: f32,
    y: f32,
    channel: usize,
    num_channels: usize) -> f32 {

    if x < 0.0 || x >= (img_width - 1) as f32 || y < 0.0 || y >= (img_height-1) as f32 {
        return 0.0;
    }

    let vals_per_line = img_width as usize * num_channels;

    let tx = x.fract();
    let ty = y.fract();
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;

    let line_lo = &pixels[y0 * vals_per_line  as usize..];
    let line_hi = &pixels[(y0+1) * vals_per_line as usize..];

    let v00 = line_lo[x0       * num_channels + channel];
    let v10 = line_lo[(x0 + 1) * num_channels + channel];
    let v01 = line_hi[x0       * num_channels + channel];
    let v11 = line_hi[(x0 + 1) * num_channels + channel];

    (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) + ty * ((1.0 - tx) * v01 + tx * v11)
}