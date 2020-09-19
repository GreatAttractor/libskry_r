//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Example showing basic usage of libskry_r.
//

extern crate skry;
use skry::defs::{ProcessingError, ProcessingPhase};
use std::io::Write;


/// Returns false on failure.
fn execute_processing_phase(phase_processor: &mut ProcessingPhase) -> bool {
    loop {
        match phase_processor.step() {
            Err(err) => match err {
                ProcessingError::NoMoreSteps => break,
                _ => { println!("Error during processing: {:?}", err); return false; }
            },
            _ => ()
        }
    }

    true
}


fn time_elapsed_str(tstart: std::time::Instant) -> String {
    let elapsed = std::time::Instant::now() - tstart;
    format!("{:.*}", 3, (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64) * 1.0e-9)
}


fn main() {
    let mut tstart = std::time::Instant::now();
    let tstart0 = tstart.clone();

    let input_file_name = "doc/sun01.avi";
    print!("Processing \"{}\"... ", input_file_name);
    let img_seq_result = skry::img_seq::ImageSequence::new_avi_video(input_file_name);
    match img_seq_result {
        Err(ref err) => { println!("Error opening file: {:?}", err); },
        Ok(_) => ()
    }

    let mut img_seq = img_seq_result.unwrap();

    print!("\nImage alignment... "); std::io::stdout().flush();

    let img_align_data;
    { // Nested scope needed to access `img_seq` mutably, here and later
        let mut img_align = skry::img_align::ImgAlignmentProc::init(
            &mut img_seq,
            skry::img_align::AlignmentMethod::Anchors(skry::img_align::AnchorConfig{
                initial_anchors: None,
                block_radius: 32,
                search_radius: 32,
                placement_brightness_threshold: 0.33
            })
        ).unwrap();

        if !execute_processing_phase(&mut img_align) {
            return;
        }

        // `__Proc` structs hold a mutable reference to the image sequence being processed,
        // so need to be enclosed in a scope (because we need to subsequently create 4
        // of these structs).
        //
        // Each `__Proc` produces a `__Data` struct, which does not hold any references,
        // so can be freely used in the main scope.

        img_align_data = img_align.get_data();
    }
    println!(" {} s", time_elapsed_str(tstart));

    print!("Quality estimation... "); std::io::stdout().flush();
    tstart = std::time::Instant::now();
    let qual_est_data;
    {
        let mut qual_est = skry::quality::QualityEstimationProc::init(&mut img_seq, &img_align_data, 40, 3);

        if !execute_processing_phase(&mut qual_est) {
            return;
        }

        qual_est_data = qual_est.get_data();
    }
    println!(" {} s", time_elapsed_str(tstart));

    print!("Reference point alignment... "); std::io::stdout().flush();
    tstart = std::time::Instant::now();

    let ref_pt_align_data: skry::ref_pt_align::RefPointAlignmentData;
    {
        let mut ref_pt_align = skry::ref_pt_align::RefPointAlignmentProc::init(
            &mut img_seq,
            &img_align_data,
            &qual_est_data,
            None,
            skry::ref_pt_align::QualityCriterion::PercentageBest(30),
            32,
            20,
            0.33,
            1.2,
            1,
            40).unwrap();

        if !execute_processing_phase(&mut ref_pt_align) {
            return;
        }

        ref_pt_align_data = ref_pt_align.get_data();
    }
    println!(" {} s", time_elapsed_str(tstart));

    print!("Image stacking... "); std::io::stdout().flush();
    tstart = std::time::Instant::now();
    let mut stacking = skry::stacking::StackingProc::init(
        &mut img_seq, &img_align_data, &ref_pt_align_data, None).unwrap();

    if !execute_processing_phase(&mut stacking) {
        return;
    }
    println!(" {} s", time_elapsed_str(tstart));

    println!("\n\nTotal time: {} s", time_elapsed_str(tstart0));
    print!("Saving \"out.tif\"... ");
    match stacking.get_image_stack().convert_pix_fmt(skry::image::PixelFormat::RGB16, None).save("out.tif", skry::image::FileType::Auto) {
        Err(err) => println!("error: {:?}", err),
        Ok(()) => println!("done.\n")
    }
}
