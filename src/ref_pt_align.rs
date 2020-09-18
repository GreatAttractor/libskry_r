//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Processing phase: reference point alignment.
//

use blk_match;
use defs::{Point, PointFlt, ProcessingError, ProcessingPhase, Rect};
use image::{DemosaicMethod, Image, ImageError, PixelFormat};
use img_align::{ImgAlignmentData};
use img_seq::{ImageSequence, SeekResult};
use quality::QualityEstimationData;
use triangulation::Triangulation;


/// Value in pixels.
const BLOCK_MATCHING_INITIAL_SEARCH_STEP: u32 = 2;

const ADDITIONAL_FIXED_PTS_PER_BORDER: usize = 4;
const ADDITIONAL_FIXED_PT_OFFSET_DIV: u32 = 4;



/// Position of a reference point in image.
#[derive(Clone, Copy)]
pub struct RefPtPosition {
    pub pos: Point,
    /// True if the quality criteria for the image are met.
    pub is_valid: bool
}


struct ReferencePoint {
    /// Index of the associated quality estimation area.
    pub qual_est_area_idx: Option<usize>,

    /// Reference block used for block matching.
    pub ref_block: Option<Image>,

    /// Positions (and their validity) in every active image.
    pub positions: Vec<RefPtPosition>,

    /// Index of the last valid position in `positions`.
    pub last_valid_pos_idx: Option<usize>,

    /// Length of the last translation vector.
    pub last_transl_vec_len: f64,
    /// Squared length of the last translation vector.
    pub last_transl_vec_sqlen: f64
}


/// Sum of reference points translation vector lengths in an image.
#[derive(Default, Clone, Copy)]
struct TVecSum {
    sum_len: f64,
    sum_sq_len: f64,
    num_terms: usize
}


/// Number of the most recent images used to keep a "sliding window" average of ref. pt. translation vector lengths.
const TVEC_SUM_NUM_IMAGES: usize = 10;


/// Selection criterion used for reference point alignment and stacking.
///
/// A "fragment" is a triangular patch.
///
#[derive(Clone, Copy)]
pub enum QualityCriterion {
    /// Percentage of best-quality fragments.
    PercentageBest(u32),

    /// Minimum relative quality (%).
    ///
    /// Only fragments with quality above specified threshold (% relative to [min,max] of
    /// the corresponding quality estimation area) will be used.
    ///
    MinRelative(f32),

    /// Number of best-quality fragments.
    NumberBest(usize)
}


#[derive(Default)]
pub struct RefPointAlignmentData {
    reference_pts: Vec<ReferencePoint>,

    /// Delaunay triangulation of the reference points
    triangulation: Triangulation,

    /// Number of valid positions of all points in all images.
    num_valid_positions: u64,

    /// Number of rejected positions of all points in all images.
    ///
    /// Concerns positions rejected by outlier testing, not by
    /// too low image quality.
    ///
    num_rejected_positions: u64
}


impl RefPointAlignmentData {
    /// Returns the number of reference points.
    pub fn get_num_ref_points(&self) -> usize {
        self.reference_pts.len()
    }


    /// Returns the final (i.e. averaged over all images) positions of reference points.
    pub fn get_final_positions(&self) -> Vec<PointFlt> {

        let mut result = Vec::<PointFlt>::with_capacity(self.reference_pts.len());
        let num_images = self.reference_pts[0].positions.len();
        for ref_pt in &self.reference_pts {
            let mut valid_pos_count = 0usize;
            result.push(PointFlt{ x: 0.0, y: 0.0 });
            let last = result.last_mut().unwrap();
            for img_idx in 0..num_images {
                // Due to how 'update_ref_pt_positions()' works, it is guaranteed that at least one element "is valid"
                if ref_pt.positions[img_idx].is_valid {
                    last.x += ref_pt.positions[img_idx].pos.x as f32;
                    last.y += ref_pt.positions[img_idx].pos.y as f32;
                    valid_pos_count += 1;
                }
            }
            last.x /= valid_pos_count as f32;
            last.y /= valid_pos_count as f32;
        }

        result
    }


    pub fn get_ref_pts_triangulation(&self) -> &Triangulation {
        &self.triangulation
    }


    //TODO: make it also available from RefPointAlignmentProc, but only for the current image (for visualization)
    /// Returns a reference point's position in the specified image and the position's "valid" flag.
    pub fn get_ref_pt_pos(&self, point_idx: usize, img_idx: usize) -> &RefPtPosition {
        &self.reference_pts[point_idx].positions[img_idx]
    }
}


#[derive(Clone)]
struct TriangleQuality {
    // Min and max sum of triangle vertices' quality in an image.
    pub qmin: f32,
    pub qmax: f32,

    /// i-th element is the sorted quality index in i-th image (quality index 0==worst).
    pub sorted_idx: Vec<usize>
}


pub struct RefPointAlignmentProc<'a> {
    img_seq: &'a mut ImageSequence,

    img_align_data: &'a ImgAlignmentData,

    qual_est_data: &'a QualityEstimationData,

    quality_criterion: QualityCriterion,

    /// Size of (square) reference blocks used for block matching.
    ///
    /// Some ref. points (near image borders) may have smaller blocks (but always square).
    ///
    ref_block_size: u32,

    /// Search radius used during block matching.
    search_radius: u32,

    /// Array of boolean flags indicating if a ref. point has been updated during the current step.
    update_flags: Vec<bool>,

    /// Contains one element for each triangle in `triangulation`.
    tri_quality: Vec<TriangleQuality>,

    is_complete: bool,

    data_returned: bool,

    /// Translation vectors of ref. points in recent images.
    ///
    /// Summary (for all ref. points) of the translation vectors between subsequent
    /// "valid" positions within the most recent `TVEC_SUM_NUM_IMAGES`.
    /// Used for clipping outliers in `update_ref_pt_positions()`.
    ///
    tvec_img_sum: [TVecSum; TVEC_SUM_NUM_IMAGES],

    /// Index in `tvec_img_sum` to store the next sum at.
    tvec_next_entry: usize,

    data: RefPointAlignmentData,
}


impl<'a> RefPointAlignmentProc<'a> {

    /// Returns image alignment data determined during processing. May be called only once.
    pub fn get_data(&mut self) -> RefPointAlignmentData {
        assert!(!self.data_returned && self.is_complete);
        self.data_returned = true;
        ::std::mem::replace(&mut self.data, RefPointAlignmentData::default())
    }


    /// Initializes reference point alignment processor.
    ///
    /// # Parameters
    ///
    /// * `img_seq` - Image sequence to process.
    ///
    /// * `first_img_ofs` - First active image's offset relative to the images' intersection.
    ///
    /// * `points` - Reference point positions; if none, points will be placed automatically.
    ///     Positions are specified within the images' intersection and must not lie outside it.
    ///
    /// * `quality_criterion` - Criterion for updating ref. point position (and later for stacking).
    ///
    /// * `ref_block_size` - Size (in pixels) of reference blocks used for block matching.
    ///
    /// * `search_radius` - Search radius (in pixels) used during block matching.
    ///
    /// * `placement_brightness_threshold` - Min. image brightness that a reference point can be placed at.
    ///     Value (from [0; 1]) is relative to the image's darkest (0.0) and brightest (1.0) pixels.
    ///
    /// * `structure_threshold` - Structure detection threshold; value of 1.2 is recommended.
    ///     The greater the value, the more local contrast is required to place a ref. point.
    ///
    /// * `structure_scale` - Corresponds to pixel size of smallest structures. Should equal 1
    ///     for optimally-sampled or undersampled images. Use higher values for oversampled (blurry) material.
    ///
    /// * `spacing` - Spacing in pixels between reference points.
    ///
    pub fn init(
        img_seq: &'a mut ImageSequence,
        img_align_data: &'a ImgAlignmentData,
        qual_est_data: &'a QualityEstimationData,
        points: Option<Vec<Point>>,
        quality_criterion: QualityCriterion,
        ref_block_size: u32,
        search_radius: u32,

        // Parameters used if `points`=None (i.e., using automatic placement of ref. points)

        placement_brightness_threshold: f32,
        structure_threshold: f32,
        structure_scale: u32,
        spacing: u32)
    -> Result<RefPointAlignmentProc<'a>, ImageError> {

        // FIXME: detect if image size < reference_block size and return an error; THE SAME goes for image alignment phase

        img_seq.seek_start();

        let intersection = qual_est_data.get_intersection();

        let mut first_img = img_seq.get_curr_img()?;

        first_img = first_img.convert_pix_fmt_of_subimage(
            PixelFormat::Mono8,
            intersection.get_pos() + img_align_data.get_image_ofs()[0],
            intersection.width,
            intersection.height,
            Some(DemosaicMethod::Simple));

        let actual_points: Vec<Point>;

        if points.is_none() {
            actual_points = qual_est_data.suggest_ref_point_positions(
                placement_brightness_threshold,
                structure_threshold,
                structure_scale,
                spacing,
                ref_block_size);
        } else {
            actual_points = points.unwrap();
        }

        let mut reference_pts = Vec::<ReferencePoint>::new();

        for point in actual_points  {
            assert!(point.x >= 0 && point.x < intersection.width as i32 &&
                    point.y >= 0 && point.y < intersection.height as i32);

            // Not initializing the reference block yet, as we do not know if the current area
            // meets the quality criteria in the current (first) image

            reference_pts.push(
                ReferencePoint{
                    qual_est_area_idx: Some(qual_est_data.get_area_idx_at_pos(&point)),
                    ref_block: None,
                    positions: vec![RefPtPosition{ pos: point, is_valid: false}; img_seq.get_active_img_count()],
                    last_valid_pos_idx: None,
                    last_transl_vec_len: 0.0,
                    last_transl_vec_sqlen: 0.0 });
        }

        RefPointAlignmentProc::append_surrounding_fixed_points(&mut reference_pts, intersection, img_seq);

        // Envelope of all reference points (including the fixed ones)
        let envelope = Rect{
            x: -(intersection.width as i32) / ADDITIONAL_FIXED_PT_OFFSET_DIV as i32,
            y: -(intersection.height as i32) / ADDITIONAL_FIXED_PT_OFFSET_DIV as i32,
            width: intersection.width + 2 * intersection.width / ADDITIONAL_FIXED_PT_OFFSET_DIV,
            height: intersection.height + 2 * intersection.height / ADDITIONAL_FIXED_PT_OFFSET_DIV };

        // Find the Delaunay triangulation of the reference points

        let initial_positions: Vec<Point> = reference_pts.iter().map(|ref p| p.positions[0].pos).collect();

        let triangulation = Triangulation::find_delaunay_triangulation(&initial_positions, &envelope);

        // The triangulation object contains 3 additional points comprising a triangle that covers all the other points.
        // These 3 points shall have fixed position and are not associated with any quality estimation area.
        // Add them to the list now and fill their position for all images.

        for i in range!(triangulation.get_vertices().len() - 3, 3) {
            reference_pts.push(RefPointAlignmentProc::create_fixed_point(triangulation.get_vertices()[i], img_seq));
        }

        let update_flags = vec![false; reference_pts.len()];

        let tri_quality = Vec::<TriangleQuality>::with_capacity(triangulation.get_triangles().len());

        let mut ref_pt_align = RefPointAlignmentProc{
                                    img_seq,
                                    img_align_data,
                                    qual_est_data,
                                    quality_criterion,
                                    ref_block_size,
                                    search_radius,
                                    update_flags,
                                    tri_quality,
                                    is_complete: false,
                                    data_returned: false,
                                    tvec_img_sum: [TVecSum::default(); TVEC_SUM_NUM_IMAGES],
                                    tvec_next_entry: 0,
                                    data: RefPointAlignmentData{ reference_pts, triangulation, num_valid_positions: 0, num_rejected_positions: 0 }
                               };

        ref_pt_align.calc_triangle_quality();

        ref_pt_align.update_ref_pt_positions(
            &first_img,
            0,
            // `first_img` is already just an intersection-sized fragment of the first image,
            // so pass a full-image "intersection" and a zero offset
            &first_img.get_img_rect(), &Point{ x: 0, y: 0 });

        Ok(ref_pt_align)
    }


    fn update_ref_pt_positions(
        &mut self,
        img: &Image,
        img_idx: usize,
        intersection: &Rect,
        img_alignment_ofs: &Point) {

        // Reminder: positions of reference points and quality estimation areas are specified
        // within the intersection of all images after alignment. Therefore all accesses
        // to the current image `img` have to take it into account and use `intersection`
        // and `img_alignment_ofs` to apply proper offsets.

        for i in &mut self.update_flags {
            *i = false;
        }

        let mut curr_step_tvec = TVecSum{ sum_len: 0.0, sum_sq_len: 0.0, num_terms: 0 };

        let num_active_imgs = self.img_seq.get_active_img_count();

//        #pragma omp parallel for
        for (tri_idx, tri) in self.data.triangulation.get_triangles().iter().enumerate() {
            // Update positions of reference points belonging to triangle `[tri_idx]` iff the sum of their quality est. areas
            // is at least the specified threshold (relative to the min and max sum)

            let mut qsum = 0.0;

            let tri_pts = [tri.v0, tri.v1, tri.v2];

            for tri_p in tri_pts.iter() {
                match self.data.reference_pts[*tri_p].qual_est_area_idx {
                    Some(qarea) => qsum += self.qual_est_data.get_area_quality(qarea, img_idx),
                    _ => ()
                }
            }

            let curr_tri_q = &self.tri_quality[tri_idx];

            let is_quality_sufficient;

            match self.quality_criterion {
                QualityCriterion::PercentageBest(percentage) => {
                    is_quality_sufficient = curr_tri_q.sorted_idx[img_idx] >= (0.01 * ((100 - percentage) * (num_active_imgs as u32)) as f32) as usize;
                },

                QualityCriterion::MinRelative(threshold) => {
                    is_quality_sufficient = qsum >= curr_tri_q.qmin + 0.01 * threshold * (curr_tri_q.qmax - curr_tri_q.qmin);
                },

                QualityCriterion::NumberBest(threshold) => {
                    is_quality_sufficient = threshold > num_active_imgs || curr_tri_q.sorted_idx[img_idx] >= num_active_imgs - threshold;
                }
            }

            for tri_p in tri_pts.iter() {

                let ref_pt = &mut self.data.reference_pts[*tri_p];

                if ref_pt.qual_est_area_idx.is_none() || self.update_flags[*tri_p] {
                    continue;
                }

                let mut found_new_valid_pos = false;

                if is_quality_sufficient {
                    let mut is_first_update = false;

                    if ref_pt.ref_block.is_none() {
                        // This is the first time this point meets the quality criteria;
                        // initialize its reference block
                        if img_idx > 0 {
                            // Point's position in the current image has not been filled in yet, do it now
                            ref_pt.positions[img_idx].pos = ref_pt.positions[img_idx-1].pos;
                        }

                        ref_pt.ref_block = Some(self.qual_est_data.create_reference_block(
                                                                       ref_pt.positions[img_idx].pos,
                                                                       self.ref_block_size));
                        is_first_update = true;
                    }

                    let current_ref_pos = ref_pt.positions[if 0 == img_idx { 0 } else { img_idx - 1 }].pos;

                    let new_pos_in_img = blk_match::find_matching_position(
                                                        current_ref_pos + intersection.get_pos() + *img_alignment_ofs,
                                                        ref_pt.ref_block.iter().next().unwrap(),
                                                        img,
                                                        self.search_radius,
                                                        BLOCK_MATCHING_INITIAL_SEARCH_STEP);

                    let new_pos = new_pos_in_img - intersection.get_pos() - *img_alignment_ofs;

                    // Additional rejection criterion: ignore the first position update if the new position is too distant.
                    // Otherwise the point would be moved too far at the very start of the ref. point alignment
                    // phase and might not recover, i.e. its subsequent position updates might be getting rejected
                    // by the additional check after the current outermost `for` loop.
                    if !is_first_update ||
                       sqr!(new_pos.x - current_ref_pos.x) + sqr!(new_pos.y - current_ref_pos.y) <= sqr!(self.ref_block_size as i32 / 3 /*TODO: make it adaptive somehow? or use the current avg. deviation*/) {

                        ref_pt.positions[img_idx].pos = new_pos;
                        ref_pt.positions[img_idx].is_valid = true;

                        match ref_pt.last_valid_pos_idx {
                            Some(lvi) => {
                                ref_pt.last_transl_vec_sqlen = Point::sqr_dist(&ref_pt.positions[img_idx].pos,
                                                                               &ref_pt.positions[lvi].pos) as f64;

                                ref_pt.last_transl_vec_len = f64::sqrt(ref_pt.last_transl_vec_sqlen);

                                curr_step_tvec.sum_len    += ref_pt.last_transl_vec_len;
                                curr_step_tvec.sum_sq_len += ref_pt.last_transl_vec_sqlen;
                                curr_step_tvec.num_terms += 1;
                            },
                            _ => ()
                        }

                        found_new_valid_pos = true;
                    }
                }

                if !found_new_valid_pos && img_idx > 0 {
                    ref_pt.positions[img_idx].pos = ref_pt.positions[img_idx-1].pos;
                }

                self.update_flags[*tri_p] = true;
            }
        }

        let mut prev_num_terms = 0usize;
        let mut prev_sum_len = 0.0f64;
        let mut prev_sum_sq_len = 0.0f64;
        for tis in &self.tvec_img_sum {
            prev_num_terms  += tis.num_terms;
            prev_sum_len    += tis.sum_len;
            prev_sum_sq_len += tis.sum_sq_len;
        }

        if curr_step_tvec.num_terms > 0 {
            let sum_len_avg: f64    = (prev_sum_len + curr_step_tvec.sum_len)       / (prev_num_terms + curr_step_tvec.num_terms) as f64;
            let sum_sq_len_avg: f64 = (prev_sum_sq_len + curr_step_tvec.sum_sq_len) / (prev_num_terms + curr_step_tvec.num_terms) as f64;

            let std_deviation: f64 =
                if sum_sq_len_avg - sqr!(sum_len_avg) >= 0.0 { f64::sqrt(sum_sq_len_avg - sqr!(sum_len_avg)) } else { 0.0 };

            // Iterate over the points found "valid" in the current step and if their current translation
            // lies too far from the "sliding window" translation average, clear their "valid" flag.
            for ref_pt in &mut self.data.reference_pts {
                if ref_pt.qual_est_area_idx.is_some() && ref_pt.positions[img_idx].is_valid &&
                   ref_pt.last_transl_vec_len > sum_len_avg + 1.5 * std_deviation {

                    ref_pt.positions[img_idx].is_valid = false;
                    ref_pt.positions[img_idx].pos = ref_pt.positions[img_idx-1].pos;

                    curr_step_tvec.sum_len -= ref_pt.last_transl_vec_len;
                    curr_step_tvec.num_terms -= 1;

                    self.data.num_rejected_positions += 1;
                } else {
                    ref_pt.last_valid_pos_idx = Some(img_idx);
                    self.data.num_valid_positions += 1;
                }
            }

            self.tvec_img_sum[self.tvec_next_entry] = curr_step_tvec;
            self.tvec_next_entry = (self.tvec_next_entry + 1) % TVEC_SUM_NUM_IMAGES;
        } else {
            for ref_pt in &mut self.data.reference_pts {
                if ref_pt.positions[img_idx].is_valid {
                    ref_pt.last_valid_pos_idx = Some(img_idx);
                    self.data.num_valid_positions += 1;
                }
            }
        }
    }


    fn calc_triangle_quality(&mut self) {
        let num_active_imgs = self.img_seq.get_active_img_count();

        #[derive(Default, Clone, Copy)]
        struct ImgIdxToQuality {
            img_idx: usize,
            quality: f32
        }

        let mut img_to_qual: Vec<ImgIdxToQuality> = vec![Default::default(); num_active_imgs];

        for tri in self.data.triangulation.get_triangles() {
            let tri_points = [&self.data.reference_pts[tri.v0],
                              &self.data.reference_pts[tri.v1],
                              &self.data.reference_pts[tri.v2]];

            let mut curr_tri_qual = TriangleQuality{
                                        qmin: ::std::f32::MAX,
                                        qmax: 0.0,
                                        sorted_idx: vec![0usize; num_active_imgs]
                                    };

            for img_idx in 0..num_active_imgs {
                let mut qsum = 0.0;

                for tri_p in tri_points.iter() {
                    match tri_p.qual_est_area_idx {
                        Some(qarea_idx) => qsum += self.qual_est_data.get_area_quality(qarea_idx, img_idx),
                        _ => () // else it is one of the fixed boundary points; does not affect triangle's quality
                    }
                }

                if qsum < curr_tri_qual.qmin {
                    curr_tri_qual.qmin = qsum;
                }
                if qsum > curr_tri_qual.qmax {
                    curr_tri_qual.qmax = qsum;
                }

                img_to_qual[img_idx] = ImgIdxToQuality{ img_idx, quality: qsum };
            }

            img_to_qual.sort_unstable_by(|x, y| x.quality.partial_cmp(&y.quality).unwrap());

            // See comment at `TriangleQuality::sorted_idx` declaration for details
            for img_idx in 0..num_active_imgs {
                curr_tri_qual.sorted_idx[img_to_qual[img_idx].img_idx] = img_idx;
            }

            self.tri_quality.push(curr_tri_qual);
        }
    }


    /// Adds a fixed reference point (not tracked during processing).
    fn create_fixed_point(pos: Point, img_seq: &ImageSequence) -> ReferencePoint {
        ReferencePoint{
            qual_est_area_idx: None,
            ref_block: None,
            positions: vec![RefPtPosition{ pos, is_valid: true }; img_seq.get_active_img_count()],
            last_valid_pos_idx: Some(0),
            last_transl_vec_len: 0.0,
            last_transl_vec_sqlen: 0.0 }
    }


    /// Adds a few fixed points along and just outside intersection's borders.
    ///
    /// This way after triangulation the near-border points will not generate skinny triangles,
    /// which would result in locally degraded stack quality.
    ///
    /// Example of triangulation without the additional points:
    ///
    /// ```
    ///  o                                                o
    ///
    ///                  +--------------+
    ///                  |  *   *   *   |
    ///                  |              |<--images' intersection
    ///                  |              |
    ///                  +--------------+
    ///
    ///
    ///
    ///                         o
    ///
    /// ```
    ///       `o` = external fixed points added by `Triangulation.find_delaunay_triangulation()`
    ///
    /// The internal near-border points (`*`) would generate skinny triangles with the upper (`o`) points.
    /// With additional fixed points:
    ///
    /// ```
    ///  o                                                o
    ///
    ///                     o   o   o
    ///
    ///                  +--------------+
    ///                o |  *   *   *   |   o
    ///                  |              |
    ///                o |              |   o
    ///
    /// ```
    ///
    fn append_surrounding_fixed_points(
        ref_points: &mut Vec<ReferencePoint>,
        intersection: &Rect,
        img_seq: &mut ImageSequence) {

        for i in 1 .. ADDITIONAL_FIXED_PTS_PER_BORDER + 1 {
            // Along top border
            ref_points.push(RefPointAlignmentProc::create_fixed_point(
                Point{ x: (i as u32 * intersection.width) as i32 / (ADDITIONAL_FIXED_PTS_PER_BORDER as i32 + 1),
                       y: -(intersection.height as i32) / ADDITIONAL_FIXED_PT_OFFSET_DIV as i32 },
                &img_seq));

            // Along bottom border
            ref_points.push(RefPointAlignmentProc::create_fixed_point(
                Point{ x: (i as u32 * intersection.width) as i32 / (ADDITIONAL_FIXED_PTS_PER_BORDER as i32 + 1),
                       y: (intersection.height + intersection.height / ADDITIONAL_FIXED_PT_OFFSET_DIV) as i32 },
                &img_seq));

            // Along left border
            ref_points.push(RefPointAlignmentProc::create_fixed_point(
                Point{ x: -(intersection.width as i32) / ADDITIONAL_FIXED_PT_OFFSET_DIV as i32,
                       y: (i as u32 * intersection.height) as i32 / (ADDITIONAL_FIXED_PTS_PER_BORDER + 1) as i32 },
                &img_seq));

            // Along right border
            ref_points.push(RefPointAlignmentProc::create_fixed_point(
                Point{ x: (intersection.width + intersection.width / ADDITIONAL_FIXED_PT_OFFSET_DIV) as i32,
                       y: (i as u32 * intersection.height) as i32 / (ADDITIONAL_FIXED_PTS_PER_BORDER + 1) as i32 },
                &img_seq));
        }
    }


    /// Makes sure that for every triangle there is at least 1 image where all 3 vertices are "valid".
    fn ensure_tris_are_valid(&mut self) {
        let num_active_imgs = self.img_seq.get_active_img_count();

        let triangles = self.data.triangulation.get_triangles();

        for tri in triangles {
            let tri_v = [tri.v0, tri.v1, tri.v2];
            let ref_pts = &mut self.data.reference_pts;

            // Best quality and associated img index where the triangle's vertices are not all "valid"
            let mut best_tri_qual = 0.0;
            let mut best_tri_qual_img_idx: Option<usize> = None;

            for img_idx in 0..num_active_imgs {
                if ref_pts[tri.v0].positions[img_idx].is_valid &&
                   ref_pts[tri.v1].positions[img_idx].is_valid &&
                   ref_pts[tri.v2].positions[img_idx].is_valid {

                    best_tri_qual_img_idx = None;
                    break;
                } else {
                    let mut tri_qual = 0.0;
                    for v in &tri_v {
                        match ref_pts[*v].qual_est_area_idx {
                            Some(qa_idx) => tri_qual += self.qual_est_data.get_area_quality(qa_idx, img_idx),
                            _ => ()
                        }
                    }
                    if tri_qual > best_tri_qual {
                        best_tri_qual = tri_qual;
                        best_tri_qual_img_idx = Some(img_idx);
                    }
                }
            }

            match best_tri_qual_img_idx {
                Some(best_idx) => {
                    // The triangle's vertices turned out not to be simultaneously "valid" in any image,
                    // which is required (in at least one image) during stacking phase.
                    //
                    // Mark them "valid" anyway in the image where their quality sum is highest.
                    for v in &tri_v {
                        ref_pts[*v].positions[best_idx].is_valid = true;
                    }
                },
                _ => ()
            }
        }
    }

}


impl<'a> ProcessingPhase for RefPointAlignmentProc<'a> {
    fn get_curr_img(&mut self) -> Result<Image, ImageError>
    {
        self.img_seq.get_curr_img()
    }


    fn step(&mut self) -> Result<(), ProcessingError> {
        match self.img_seq.seek_next() {
            Err(SeekResult::NoMoreImages) => {
                self.ensure_tris_are_valid();
                self.is_complete = true;
                return Err(ProcessingError::NoMoreSteps);
            },
            Ok(()) => ()
        }

        let img_idx = self.img_seq.get_curr_img_idx_within_active_subset();

        let mut img = self.img_seq.get_curr_img()?;
        if img.get_pixel_format() != PixelFormat::Mono8 {
            img = img.convert_pix_fmt(PixelFormat::Mono8, Some(DemosaicMethod::Simple));
        }

        self.update_ref_pt_positions(
            &img, img_idx,
            &self.img_align_data.get_intersection(),
            &self.img_align_data.get_image_ofs()[img_idx]);

        Ok(())
    }
}
