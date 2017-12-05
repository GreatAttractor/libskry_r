// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Processing phase: quality estimation.
//

use blk_match;
use defs::{Point, ProcessingError, ProcessingPhase, Rect, WHITE_8BIT};
use filters;
use image::{DemosaicMethod, Image, ImageError, PixelFormat};
use img_align::ImgAlignmentData;
use img_seq::{ImageSequence, SeekResult};
use std::cmp::{min, max};
use utils;


/// Summary of area's quality in all images.
#[derive(Copy, Clone)]
pub struct AreaQualitySummary {
    pub min: f32,
    pub max: f32,
    pub avg: f32,
    
    pub best_img_idx: usize
}


/// Quality estimation area.
struct QualEstArea {
    /// Area's boundaries within the images' intersection.
    pub rect: Rect, 

    /// Contains a fragment of the image in which the estimation area has the highest quality (out of all sequence's images).
    ///
    /// The fragment is a rectangle 3x wider and higher than `rect`, with `rect` in the middle;
    /// for areas near the border of intersection it may be smaller.
    /// The field is assigned by `on_final_step()`.
    ///
    pub ref_block: Option<Image>,

    /// Position of `ref_block` within the images' intersection.
    pub ref_block_pos: Point,
    
    pub summary: AreaQualitySummary
}


#[derive(Default)]
struct OverallQuality {
    /// Average area quality.
    pub area_avg: f32,
    /// Max. avg. quality of all areas.
    pub area_max_avg: f32,
    /// Min. non-zero area quality.
    pub area_min_nonzero_avg: f32,

    /// Index of the best-quality image.
    pub image_best_img_idx: usize,
    /// Best image quality.
    pub image_best_quality: f32
}


/// Contains results of processing performed by `QualityEstimationProc`.
#[derive(Default)]
pub struct QualityEstimationData {
    /// Images' intersection.
    intersection: Rect,
    
    /// Number of quality est. areas that span the images' intersection horizontally.
    num_areas_horz: usize,
    
    /// Definitions of quality estimation areas.
    area_defs: Vec<QualEstArea>,
    
    // Brightness of quality estimation areas' reference blocks.
    min_ref_block_brightness: u8,
    max_ref_block_brightness: u8,
    
    /// Most areas are squares with sides of `area_size` length (border areas may be smaller).
    area_size: u32,
    
    /// Array of the areas' quality in all images.
    ///
    /// Each element corresponds to an active image in the image sequence,
    /// and is a vector containing one element for each quality estimation area.
    ///
    area_quality: Vec<Vec<f32>>,
    
    overall: OverallQuality,
    
    /// Overall quality of images in the image sequence.
    img_quality: Vec<f32>
}


impl QualityEstimationData {
    
    pub fn get_intersection(&self) -> &Rect { &self.intersection }
    
    pub fn get_num_areas(&self) -> usize {
        self.area_defs.len()
    }
    
    
    pub fn get_min_nonzero_avg_area_quality(&self) -> f32 {
        return self.overall.area_min_nonzero_avg
    }
    
    
    pub fn get_overall_avg_area_quality(&self) -> f32 {
        return self.overall.area_avg
    }
    
    
    pub fn get_best_img_idx(&self) -> usize {
        self.overall.image_best_img_idx
    }
        
    
    /// Returns area quality in the specified image.
    pub fn get_area_quality(&self, area_idx: usize, img_idx: usize) -> f32 	{
        self.area_quality[img_idx][area_idx]
    }


    pub fn get_best_avg_area_quality(&self) -> f32 {
        self.overall.area_max_avg
    }
    
    
    pub fn get_qual_est_area_center(&self, area_idx: usize) -> Point {
        let arect = &self.area_defs[area_idx].rect;
        
        Point { x: arect.x + arect.width as i32/2,
                y: arect.y + arect.height as i32/2 }
    }

    
    /// Returns suggested reference point positions.
    ///
    /// # Parameters
    ///
    /// * `brightness_threshold` - Min. image brightness that a ref. point can be placed at;
    ///                            value (from [0; 1]) is relative to the darkest (0.0)
    ///                            and brightest (1.0) pixels.
    /// * `structure_threshold` - Structure detection threshold; value of 1.2 is recommended.
    ///                           The greater the value, the more local contrast is required
    ///                           to place a reference point.
    /// * `structure_scale` - Corresponds to pixel size of the smallest structures. Should equal 1
    ///                       for optimally-sampled or undersampled images. Use higher values
    ///                       for oversampled (blurry) material.
    /// * `spacing` - Spacing in pixels between reference points.
    /// * `ref_block_size` - Size of reference blocks used for block matching.
    ///
    pub fn suggest_ref_point_positions(
        &self,
        brightness_threshold: f32,
        structure_threshold: f32,
        structure_scale: u32,
        spacing: u32,
        ref_block_size: u32) -> Vec<Point> {
            
        let grid_step = spacing;

        let num_grid_cols = self.intersection.width / grid_step as u32;
        let num_grid_rows = self.intersection.height / grid_step as u32;

        let mut result = Vec::<Point>::new();

        // Uniform grid covering the images' intersection, each element may contain one reference point
        let mut grid: Vec<Option<Point>> = vec![None; (num_grid_cols * num_grid_rows) as usize];

        macro_rules! cell_idx
        { ($row:expr, $col:expr) => { ($col + $row * num_grid_cols as i32) as usize } };

        macro_rules! contains_ref_pt {
            ($row:expr, $col:expr) => {
                if $row < 0 || $row >= num_grid_rows as i32 || $col < 0 || $col >= num_grid_cols as i32 {
                    false
                } else {
                    grid[cell_idx!($row, $col)].is_some()
                }
            }
        };

        for grid_row in 0..num_grid_rows as i32 {
            for grid_col in 0..num_grid_cols as i32 {
                let search_step = ref_block_size as i32 / 2;
                let mut best_fitness = 0.0f64;
                let mut best_pos = Point::default();
    
                // Do not try to place ref. points too close to images' intersection border
                let ystart = if grid_row > 0 { 0i32 } else { ref_block_size as i32 / 2 };
                let yend = if grid_row <= num_grid_rows as i32 - 2 { grid_step as i32 }
                           else { self.intersection.height as i32 - (num_grid_rows as i32 - 1) * grid_step as i32 - ref_block_size as i32 / 2 };

                let xstart = if grid_col > 0 { 0 } else { ref_block_size as i32 / 2 };
                let xend = if grid_col <= num_grid_cols as i32 - 2 { grid_step as i32 }
                           else { self.intersection.width as i32 - (num_grid_cols as i32 - 1) * grid_step as i32 - ref_block_size as i32 / 2 };

                { // Sub-scope needed to temporarily store refs. to `grid`'s elements in `neighbor_cell_points`
                  
                    let mut neighbor_cell_points = Vec::<Option<&Point>>::with_capacity(8);
        
                    // Check the 8 neighboring cells for already existing ref. points
                    for d_row in -1i32..2 {
                        for d_col in -1i32..2 {
                            if d_row != 0 || d_col != 0 {
                                if contains_ref_pt!(grid_row + d_row, grid_col + d_col) {
                                    neighbor_cell_points.push(
                                        Some(grid[cell_idx!(grid_row + d_row, grid_col + d_col)].iter().next().unwrap())
                                    );
                                }
                            }
                        }
                    }
        
                    let mut y = ystart;
                    while y < yend {
                        let mut x = xstart;
                        while x < xend {
                            let curr_pos = Point{ x: grid_col * grid_step as i32 + x, y: grid_row * grid_step as i32 + y };
    
                            // Do not assess a location if there are already reference points
                            // in neighboring grid cells closer than `spacing`
                            let mut too_close_to_neighbor = false;
    
                            for npoint in &neighbor_cell_points {
                                if sqr!(curr_pos.x - npoint.unwrap().x) + sqr!(curr_pos.y - npoint.unwrap().y) < sqr!(spacing as i32) {
                                    too_close_to_neighbor = true;
                                    break;
                                }
                            }
        
                            if !too_close_to_neighbor {
                                let fitness = self.assess_ref_pt_location(curr_pos, ref_block_size, structure_scale, brightness_threshold);
                                if fitness > best_fitness {
                                    best_fitness = fitness;
                                    best_pos = curr_pos;
                                }
                            }
    
                            x += search_step;
                        }
                        
                        y += search_step;
                    }
                } // Sub-scope ends, now `grid` can be updated
    
                if best_fitness >= structure_threshold as f64 {
                    grid[cell_idx!(grid_row, grid_col)] = Some(best_pos);
                    result.push(best_pos);
                }
            }
        }

        result
    }

    
    /// Checks if a position has some above-background and below-white neighboring pixel values.
    ///
    /// Returns `true` if the neighborhood of `pos` contains enough pixels above the background threshold
    /// and not too many white ones (e.g. belonging to an overexposed solar disk).
    ///
    /// # Parameters
    ///
    /// * `q_area` - Quality est. area containing `pos`.
    /// * `pos` - Position within images' intersection.
    /// * `brightness_threshold` - Min. image brightness that a ref. point can be placed at;
    ///                            value (from [0; 1]) is relative to the darkest (0.0)
    ///                            and brightest (1.0) pixels.
    ///
    fn background_threshold_met(&self,
                                q_area: &QualEstArea,
                                pos: Point,
                                neighborhood_size: usize,
                                brightness_threshold: f32) -> bool {
                                    
        let mut is_neighb_brightness_sufficient = false;
        let mut non_white_px_count = 0usize;
        
        let ref_block: &Image = q_area.ref_block.iter().next().unwrap();
    
        for ny in max(pos.y - neighborhood_size as i32, q_area.ref_block_pos.y)
                  ..
                  min(pos.y + neighborhood_size as i32, q_area.ref_block_pos.y + ref_block.get_height() as i32 - 1) + 1 {
                      
            let line = ref_block.get_line_raw((ny - q_area.ref_block_pos.y) as u32);

            for nx in max(pos.x - neighborhood_size as i32, q_area.ref_block_pos.x)
                      ..
                      min(pos.x + neighborhood_size as i32, q_area.ref_block_pos.x + ref_block.get_width() as i32 - 1) + 1 {
                          
                let val = line[(nx - q_area.ref_block_pos.x) as usize];
                if val >= self.min_ref_block_brightness +
                    (brightness_threshold * (self.max_ref_block_brightness - self.min_ref_block_brightness) as f32) as u8 {
                              
                    is_neighb_brightness_sufficient = true;
                }
    
                if val < WHITE_8BIT { non_white_px_count += 1; }
            }
        }
    
        // Require at least 1/3rd of neighborhood's pixels to be non-white
        let is_outside_white_disc = non_white_px_count > sqr!(2*neighborhood_size + 1) / 3;
    
        is_neighb_brightness_sufficient && is_outside_white_disc
    }
    
    
    /// Assesses a potential location of a reference point and returns its quality (the higher, the better).
    ///
    /// Uses two criteria to check if `pos` is appropriate for block matching (performed during reference point
    /// alignment):
    ///
    ///   * distribution of pixel brightness gradients around `pos`
    ///   * variability of pixel differences during block matching around `pos`
    ///
    ///
    /// # Parameters
    ///
    /// * `pos` - Position within images' intersection.
    /// * `block_size` - Size of reference block to use.
    /// * `structure_scale` - Corresponds to pixel size of the smallest structures.
    /// * `brightness_threshold` - Min. image brightness that a ref. point can be placed at;
    ///                            value (from [0; 1]) is relative to the darkest (0.0)
    ///                            and brightest (1.0) pixels.
    ///
    fn assess_ref_pt_location(&self,
                              pos: Point,
                              block_size: u32,	
                              structure_scale: u32,
                              brightness_threshold: f32) -> f64 {

        let qarea = &self.area_defs[self.get_area_idx_at_pos(&pos)];
        let qarea_ref_block: &Image = qarea.ref_block.iter().next().unwrap();

        if !self.background_threshold_met(qarea, pos, 5 /*TODO: make it configurable?*/, brightness_threshold) {
            return 0.0;
        }
    
        // See the function's header comment for details on this check
        if !utils::assess_gradients_for_block_matching(
            qarea_ref_block,
            Point { x: pos.x - qarea.ref_block_pos.x,
                    y: pos.y - qarea.ref_block_pos.y },
            32) { // This cannot be too small (histogram would be too sparse), perhaps should depend on ref. pt. spacing
            
            return 0.0;
        }

        let ref_block = qarea_ref_block.get_fragment_copy(
            Point{ x: pos.x - block_size as i32/2 - qarea.ref_block_pos.x,
                   y: pos.y - block_size as i32/2 - qarea.ref_block_pos.y },
            block_size, block_size, false);
    
        // A good ref. point location should have a significant difference between sums of pixel differences
        // obtained in square "shells" centered at `pos`, having radii of 1x and 2x`structure_scale`
    
        let sum_diffs_1 = get_sum_diffs_in_shell(
                            qarea_ref_block,
                            &ref_block,
                            Point { x: pos.x - qarea.ref_block_pos.x,
                                    y: pos.y - qarea.ref_block_pos.y },
                            structure_scale)
            / structure_scale as u64;

        let sum_diffs_2 = get_sum_diffs_in_shell(
                            &qarea_ref_block,
                            &ref_block,
                            Point { x: pos.x - qarea.ref_block_pos.x,
                                    y: pos.y - qarea.ref_block_pos.y },
                            2*structure_scale)
            / (2*structure_scale) as u64;
    
        if sum_diffs_1 > 0 { sum_diffs_2 as f64 / sum_diffs_1 as f64 } else { 0.0 }
    }
    
    
    /// Returns a composite image consisting of the best fragments of all frames.
    pub fn get_best_fragments_img(&self) -> Image {
        let mut result = Image::new(self.intersection.width,
                                    self.intersection.height,
                                    PixelFormat::Mono8, None, false);

        for area in &self.area_defs {
            area.ref_block.iter().next().unwrap().convert_pix_fmt_of_subimage_into(
                &mut result,
                Point{ x: area.rect.x - area.ref_block_pos.x,
                       y: area.rect.y - area.ref_block_pos.y },
                Point{ x: area.rect.x, y: area.rect.y },
                area.rect.width, area.rect.height,
                Some(DemosaicMethod::Simple));
        }
    
        result
    }
    
    
    pub fn get_avg_area_quality(&self, area_idx: usize) -> f32 {
        self.area_defs[area_idx].summary.avg }
    
    
    /// Returns min, max, avg quality of the specified area.
    pub fn get_area_quality_summary(&self, area_idx: usize) -> AreaQualitySummary {
        self.area_defs[area_idx].summary
    }
    
    
    /// Returns a square image to be used as reference block for reference point alignment.
    ///
    /// # Parameters
    ///
    /// * `pos` - Center of the reference block (within images' intersection).
    /// * `blk_size` - Desired width & height; the result may be smaller than this (but always a square). 
    ///
    pub fn create_reference_block(&self, pos: Point, blk_size: u32) -> Image {
        let area = &self.area_defs[self.get_area_idx_at_pos(&pos)];
        let ref_block: &Image = area.ref_block.iter().next().unwrap();
    
        let area_refb_w = ref_block.get_width();
        let area_refb_h = ref_block.get_height();
    
        // Caller is requesting a square block of `blk_size`. We need to copy it from `area.ref_block`.
        // Determine the maximum size of square we can return (the square must be centered on `pos`
        // and fit in `area.ref_block`):
        // 
        // +----------images' intersection-------------...
        // |
        // |    area.ref_block_pos
        // |                 *-----------area.ref_block--------+
        // .                 |                                 |
        // .                 |                    +============+
        // .                 |                    |            |
        //                   |                    |            |
        //                   |        result_size{|     *pos   |
        //                   |                    |            |
        //                   |                    |            |
        //                   |                    +============+
        //                   |                                 |
        //                   ...

        let mut result_size = blk_size;
        result_size = min(result_size, 2 * (pos.x - area.ref_block_pos.x) as u32);
        result_size = min(result_size, 2 * (pos.y - area.ref_block_pos.y) as u32);
        result_size = min(result_size, 2 * (area.ref_block_pos.x + area_refb_w as i32 - pos.x) as u32);
        result_size = min(result_size, 2 * (area.ref_block_pos.y + area_refb_h as i32 - pos.y) as u32);

        ref_block.get_fragment_copy(Point{ x: pos.x - area.ref_block_pos.x - result_size as i32/2,
                                           y: pos.y - area.ref_block_pos.y - result_size as i32/2 },
                                    result_size, result_size, false)
    }
    
    
    /// Returns the index of quality estimation area at the specified position in images' intersection.
    pub fn get_area_idx_at_pos(&self, pos: &Point) -> usize {
        
        // See init() for how the estimation areas are placed within the images' intersection

        let col = pos.x as usize / self.area_size as usize;
        let row = pos.y as usize / self.area_size as usize;
    
        row * self.num_areas_horz + col
    }
} 


/// Performs quality estimation of images.
///
/// Determines quality of image fragments and overall quality of whole images.
/// Quality is the sum of differences between an image (or a fragment) and its blurred version.
/// In other words, sum of values of the high-frequency component. The sum is normalized
/// by dividing by the number of pixels.
///
pub struct QualityEstimationProc<'a> {
    img_seq: &'a mut ImageSequence,
    
    img_align_data: &'a ImgAlignmentData,
    
    is_estimation_complete: bool,
    
    data_returned: bool,
    
    box_blur_radius: u32,

    first_step_complete: bool,

    data: QualityEstimationData
}


impl<'a> QualityEstimationProc<'a> {
    /// Returns image quality data determined during processing. May be called only once.
    pub fn get_data(&mut self) -> QualityEstimationData {
        assert!(!self.data_returned && self.is_complete());
        self.data_returned = true;
        ::std::mem::replace(&mut self.data, QualityEstimationData::default())
    }

    
    /// # Parameters
    ///
    /// * `estimation_area_size` - Aligned image sequence will be divided into areas of this size for quality estimation.
    /// * `detail_scale` - Corresponds to box blur radius used for quality estimation.
    /// 
    pub fn init(img_seq: &'a mut ImageSequence,
                img_align_data: &'a ImgAlignmentData,
                estimation_area_size: u32,
                detail_scale: u32) -> QualityEstimationProc<'a> {
                    
        assert!(estimation_area_size > 0);
        assert!(detail_scale > 0);
    
        let intersection = img_align_data.get_intersection();
        let i_width = intersection.width;
        let i_height = intersection.height;    

        // Divide the aligned images' intersection into quality estimation areas.
        // Each area is a square of `estimation_area_size` pixels. If there are left-overs,
        // assign them to appropriately smaller areas at the intersection's right and bottom border:
        //
        // Example: num_areas_horz = 4, num_areas_vert = 3
        // 
        //  +----+----+----+--+
        //  |    |    |    |  |
        //  |    |    |    |  |
        //  +----+----+----+--+
        //  |    |    |    |  |
        //  |    |    |    |  |
        //  +----+----+----+--+
        //  |    |    |    |  | 
        //  +----+----+----+--+
        //
        //

        // Number of areas in the aligned images' intersection in horizontal and vertical direction
        let num_areas_horz = updiv!(i_width, estimation_area_size) as usize;

        let width_rem = i_width % estimation_area_size;
        let height_rem = i_height % estimation_area_size;
    
        let mut area_defs = Vec::<QualEstArea>::new();
    
        for y in 0 .. i_height/estimation_area_size {
            for x in 0 .. i_width/estimation_area_size {
                area_defs.push(QualEstArea{ rect: Rect{ x: (x * estimation_area_size) as i32,
                                                        y: (y * estimation_area_size) as i32,
                                                        width: estimation_area_size,
                                                        height: estimation_area_size },
                                            ref_block: None,
                                            ref_block_pos: Point::default(),
                                            summary: AreaQualitySummary{ min: ::std::f32::MAX,
                                                                         max: 0.0,
                                                                         avg: 0.0,
                                                                         best_img_idx: 0 }
                });
            }
    
            // Additional smaller area on the right
            if width_rem != 0 {
                area_defs.push(QualEstArea{ rect: Rect{ x: (i_width - width_rem) as i32,
                                                        y: (y * estimation_area_size) as i32,
                                                        width: width_rem,
                                                        height: estimation_area_size },
                                             ref_block: None,
                                             ref_block_pos: Point::default(),
                                             summary: AreaQualitySummary{ min: ::std::f32::MAX,
                                                                          max: 0.0,
                                                                          avg: 0.0,
                                                                          best_img_idx: 0 }
                });
            }
        }
    
        // Row of additional smaller areas at the bottom
        if height_rem != 0 {
            for x in 0 .. i_width/estimation_area_size {
                area_defs.push(QualEstArea{ rect: Rect{ x: (x * estimation_area_size) as i32,
                                                        y: (i_height - height_rem) as i32,
                                                        width: estimation_area_size,
                                                        height: height_rem },
                                            ref_block: None,
                                            ref_block_pos: Point::default(),
                                            summary: AreaQualitySummary{ min: ::std::f32::MAX,
                                                                         max: 0.0,
                                                                         avg: 0.0,
                                                                         best_img_idx: 0 }
                });

            }
    
            if width_rem != 0 {
                area_defs.push(QualEstArea{ rect: Rect{ x: (i_width - width_rem) as i32,
                                                        y: (i_height - height_rem) as i32,
                                                        width: width_rem,
                                                        height: height_rem },
                                            ref_block: None,
                                            ref_block_pos: Point::default(),
                                            summary: AreaQualitySummary{ min: ::std::f32::MAX,
                                                                         max: 0.0,
                                                                         avg: 0.0,
                                                                         best_img_idx: 0 }
                });                                            
            }
        }
        
        img_seq.seek_start();

        QualityEstimationProc::<'a>{
            img_seq,
            img_align_data,
            is_estimation_complete: false,
            data_returned: false,
            box_blur_radius: detail_scale,
            first_step_complete: false,
            data: QualityEstimationData{
                intersection,
                num_areas_horz,
                area_defs,
                min_ref_block_brightness: u8::max_value(),
                max_ref_block_brightness: u8::min_value(),
                area_size: estimation_area_size,
                overall: OverallQuality::default(),
                area_quality: vec![],
                img_quality: vec![]
           }
        }
    }
    
    
    pub fn is_complete(&self) -> bool { self.is_estimation_complete }
    
    
    /// Creates reference blocks for the quality estimation areas, using images where the areas have the best quality.
    fn create_reference_blocks(&mut self) -> Result<(), ImageError> {
        let intrs_ofs = Point{ x: self.data.intersection.x, y: self.data.intersection.y };
    
        self.img_seq.seek_start();
        loop {
            let curr_img_idx = self.img_seq.get_curr_img_idx_within_active_subset();
            
            let mut curr_img_opt: Option<Image> = None;

            for qarea in &mut self.data.area_defs {
                if qarea.summary.best_img_idx == curr_img_idx {
                    if curr_img_opt.is_none() {
                        curr_img_opt = Some(try!(self.img_seq.get_curr_img()));
                        if curr_img_opt.iter().next().unwrap().get_pixel_format() != PixelFormat::Mono8 {
                            curr_img_opt = Some(curr_img_opt.iter().next().unwrap().convert_pix_fmt(PixelFormat::Mono8, Some(DemosaicMethod::Simple)));
                        }
                    }
                    let curr_img: &Image = curr_img_opt.iter().next().unwrap();
                    
                    let curr_img_ofs = self.img_align_data.get_image_ofs()[curr_img_idx];

                    // Position of `qarea` in `curr_img`
                    let curr_area_pos =
                        Point{ x: intrs_ofs.x + curr_img_ofs.x + qarea.rect.x,
                               y: intrs_ofs.y + curr_img_ofs.y + qarea.rect.y };
    
                    let asize = self.data.area_size as i32;
    
                    // The desired size of ref. block is `3*asize` in width and height;
                    // need to make sure it fits in `curr_img`
                    let ifrag_x = max(0, curr_area_pos.x + qarea.rect.width as i32/2 - 3*asize/2);
                    let ifrag_y = max(0, curr_area_pos.y + qarea.rect.height as i32/2 - 3*asize/2);
                    let ifrag_width  = min(curr_img.get_width() as i32 - ifrag_x, 3*asize);
                    let ifrag_height = min(curr_img.get_height() as i32 - ifrag_y, 3*asize);

                    qarea.ref_block_pos = Point{ x: ifrag_x - intrs_ofs.x - curr_img_ofs.x,
                                                 y: ifrag_y - intrs_ofs.y - curr_img_ofs.y };

                    qarea.ref_block = Some(curr_img.get_fragment_copy(Point{ x: ifrag_x, y: ifrag_y },
                                                                      ifrag_width as u32, ifrag_height as u32,
                                                                      false)); 
                }
            }

            match self.img_seq.seek_next() {
                Err(SeekResult::NoMoreImages) => break,
                _ => { }
            }
        }

        Ok(())
    }
    
    
    fn on_final_step(&mut self) -> Result<(), ImageError> {
        let num_active_imgs = self.img_seq.get_active_img_count();

        try!(self.create_reference_blocks());    
    
        self.data.overall.area_max_avg = 0.0;
        self.data.overall.area_min_nonzero_avg = ::std::f32::MAX;
    
        let mut overall_sum = 0.0f64;

        for i in 0..self.data.area_defs.len() {
            let mut quality_sum = 0.0f32;

            for image_qual in &self.data.area_quality {
                quality_sum += image_qual[i];
                overall_sum += image_qual[i] as f64;
            }
    
            let qavg = quality_sum / num_active_imgs as f32;
            self.data.area_defs[i].summary.avg = qavg;
    
            if qavg > self.data.overall.area_max_avg {
                self.data.overall.area_max_avg = qavg;
            }

            if qavg > 0.0 && qavg < self.data.overall.area_min_nonzero_avg {
                self.data.overall.area_min_nonzero_avg = qavg;
            }

            let (bmin, bmax) = utils::find_min_max_brightness(self.data.area_defs[i].ref_block.iter().next().unwrap());

            if bmin < self.data.min_ref_block_brightness {
                self.data.min_ref_block_brightness = bmin;
            }
            if bmax > self.data.max_ref_block_brightness {
                self.data.max_ref_block_brightness = bmax;
            }
        }
    
        self.data.overall.area_avg = overall_sum as f32 / (self.data.area_defs.len() * num_active_imgs) as f32;
        self.is_estimation_complete = true;

        Ok(())
    }
    
}


impl<'a> ProcessingPhase for QualityEstimationProc<'a> {
    fn get_curr_img(&mut self) -> Result<Image, ImageError> {
        self.img_seq.get_curr_img()
    }

    
    fn step(&mut self) -> Result<(), ProcessingError> {
        if self.first_step_complete {
            match self.img_seq.seek_next() {
                Err(SeekResult::NoMoreImages) =>
                    match self.on_final_step() {
                        Err(err) => return Err(ProcessingError::ImageError(err)),
                        _ => return Err(ProcessingError::NoMoreSteps)
                    },
                    
                Ok(()) => { }
            }
        }
    
        let curr_img_idx = self.img_seq.get_curr_img_idx_within_active_subset();

        let mut curr_img = try!(self.img_seq.get_curr_img());
        if curr_img.get_pixel_format() != PixelFormat::Mono8 {
            curr_img = curr_img.convert_pix_fmt(PixelFormat::Mono8, Some(DemosaicMethod::Simple));
        }

        // Current image's quality values for all estimation areas
        let mut curr_img_area_quality = Vec::<f32>::new();

        let mut curr_img_qual = 0.0f32;
    
        let alignment_ofs = self.img_align_data.get_image_ofs()[curr_img_idx];
            
        let intersection = self.img_align_data.get_intersection();

//        #pragma omp parallel for \
//         reduction(+:curr_img_qual)
        for area in &mut self.data.area_defs {
            let aqual = filters::estimate_quality(
                curr_img.get_mono8_pixels_from(Point{ x: area.rect.x + intersection.x + alignment_ofs.x,
                                                      y: area.rect.y + intersection.y + alignment_ofs.y }),
                area.rect.width,
                area.rect.height,
                curr_img.get_bytes_per_line(),
                self.box_blur_radius);
    
            curr_img_qual += aqual;
            curr_img_area_quality.push(aqual);
            if aqual > area.summary.max {
                area.summary.max = aqual;
                area.summary.best_img_idx = curr_img_idx;
            }
            if aqual < area.summary.min {
                area.summary.min = aqual;
            }
        }

        self.data.area_quality.push(curr_img_area_quality);
        self.data.img_quality.push(curr_img_qual);

        if curr_img_qual > self.data.overall.image_best_quality {
            self.data.overall.image_best_quality = curr_img_qual;
            self.data.overall.image_best_img_idx = curr_img_idx;
        }
        
        if !self.first_step_complete {
            self.first_step_complete = true;
        }

        Ok(())
    }
}


/// Returns the intersection of `rect1` and `rect2`.
///
/// Result's position is relative to `rect2`s origin. E.g., if `rect2` lies entirely
/// within `rect1`, the result is `{0, 0, rect2.width, rect2.height}`.
/// If the intersection is empty, result's width and height will be zero.
///
fn find_rect_intersection(rect1: Rect, rect2: Rect) -> Rect {

    let x = max(0, rect1.x - rect2.x);
    let y = max(0, rect1.y - rect2.y);

    let xmax = min(rect2.x + rect2.width as i32, rect1.x + rect1.width as i32);
    let ymax = min(rect2.y + rect2.height as i32, rect1.y + rect1.height as i32);

    Rect{ x, y, width: (xmax - rect2.x - x) as u32, height: (ymax - rect2.y - y) as u32 }
}


/// Calculates the sum of squared differences of pixel values between `img` and `ref_block` positioned in a shell.
///
/// For the calculation, `ref_block`'s center is positioned on every `img`'s pixel belonging to a square shell
/// (1-pixel thick) with the specified center and radius.
///
/// # Parameters
///
/// * `radius` - Radius of a square shell around `cmp_pos`; must be > 0.
/// * `cmp_pos` - Center of the shell (relative to `img`).
///
fn get_sum_diffs_in_shell(img: &Image, ref_block: &Image, cmp_pos: Point, radius: u32) -> u64 {
    assert!(radius > 0);

    let mut shell_pos = Point::default(); // Comparison position on the shell (relative to `img`)
    let mut cmp_rect: Rect; // Extents of `ref_block` relative to `img`

    let mut sum_diffs = 0u64;

    let blk_w = ref_block.get_width();
    let blk_h = ref_block.get_height();

    macro_rules! calc_sums{
        () => {
            cmp_rect = Rect{ x: shell_pos.x - blk_w as i32/2,
                             y: shell_pos.y - blk_h as i32/2,    
                             width: blk_w as u32,                
                             height: blk_h as u32 };             
                                                              
            sum_diffs += blk_match::calc_sum_of_squared_diffs(
                &img, &ref_block,         
                &shell_pos,             
                &find_rect_intersection(img.get_img_rect(), cmp_rect));
        }
    };


    for i in -(radius as i32) .. radius as i32 + 1 {
        // Top border
        shell_pos.x = cmp_pos.x + i;
        shell_pos.y = cmp_pos.y - radius as i32;

        calc_sums!();

        // Bottom border
        shell_pos.x = cmp_pos.x + i;
        shell_pos.y = cmp_pos.y + radius as i32;

        calc_sums!();
    }

    for i in -(radius as i32) + 1 .. radius as i32 {
        // Left border
        shell_pos.x = cmp_pos.x - radius as i32;
        shell_pos.y = cmp_pos.y - i;

        calc_sums!();

        // Right border
        shell_pos.x = cmp_pos.x + radius as i32;
        shell_pos.y = cmp_pos.y + i;

        calc_sums!();
    }

    sum_diffs
}