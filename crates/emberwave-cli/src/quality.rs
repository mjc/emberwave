//! `emberwave-cli quality`: measure a stream against the pictures it was
//! encoded from, in PSNR and in the error the encoder's rate control
//! minimises.

use std::path::PathBuf;

use emberwave_core::bitstream::pwv1::Pwv1File;
use emberwave_core::bitstream::{FrameAssembler, FrameLayout};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{CoefficientError, coefficient_error, coefficient_error_u16};

use crate::compare::format_db;
use crate::{Result, chroma_name, read_file, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The pictures the stream was encoded from: an 8-bit or 10-bit Y4M file
    source: PathBuf,

    /// The stream: a PWV1 file from any encoder, upstream's too
    stream: PathBuf,

    /// Measure the first N frames; the files may have more. Without it, they
    /// must have as many as each other
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    frames: Option<u64>,
}

pub fn run(args: &Args) -> Result<()> {
    let bytes = read_file(&args.stream)?;
    let file = Pwv1File::parse(&bytes).map_err(|error| in_stream(args, error))?;
    let stream = &file.header().stream;
    let layout = FrameLayout::new(stream.width, stream.height, stream.chroma)?;
    let mut source = y4m::open(&args.source)?;
    let header = *source.header();
    let size = (stream.width as usize, stream.height as usize, stream.chroma);
    if (header.width, header.height, header.chroma) != size {
        return Err(format!(
            "the pictures are {}x{} {} and the stream {}x{} {}",
            header.width,
            header.height,
            chroma_name(header.chroma),
            stream.width,
            stream.height,
            chroma_name(stream.chroma)
        )
        .into());
    }
    let stream_depth = file.header().reference_bit_depth;
    if stream_depth != 0 && stream_depth != header.bit_depth.bits() {
        return Err(format!(
            "the pictures are {}-bit and the stream reference bit depth is {}-bit",
            header.bit_depth.bits(),
            stream_depth
        )
        .into());
    }

    let limit = args.frames.unwrap_or(u64::MAX);
    let mut assembler = FrameAssembler::new(layout);
    let mut frames = file.frames();
    let mut count = 0usize;
    let mut bytes_total = 0usize;
    // Per plane, over all frames: squared differences and samples.
    let mut squared_total = [0u64; 3];
    let mut samples_total = [0u64; 3];
    let mut error_total = CoefficientError::default();
    while (count as u64) < limit {
        let picture = source
            .next_frame()
            .map_err(|error| format!("{}: {error}", args.source.display()))?;
        let frame = frames
            .next()
            .transpose()
            .map_err(|error| in_stream(args, error))?;
        let (picture, frame) = match (picture, frame) {
            (Some(picture), Some(frame)) => (picture, frame),
            (None, None) => break,
            (None, Some(_)) | (Some(_), None) => {
                return Err(format!(
                    "the pictures and the stream have different numbers of frames: one ends after \
                     {count}; --frames measures the first ones"
                )
                .into());
            }
        };
        assembler.clear();
        assembler
            .push_packet(frame)
            .map_err(|error| in_frame(args, count, error))?;
        if !assembler.is_ready(false) {
            return Err(in_frame(
                args,
                count,
                format!(
                    "incomplete: {} of {} blocks",
                    assembler.received_blocks(),
                    assembler.total_blocks()
                ),
            ));
        }
        let view = assembler.frame();
        let (error, decoded_squared, decoded_samples) = match header.bit_depth {
            y4m::BitDepth::Eight => {
                let planes = [&picture[0][..], &picture[1][..], &picture[2][..]];
                let error = coefficient_error(planes, &view)
                    .map_err(|error| in_frame(args, count, error))?;
                let decoded = decode_frame(&view).map_err(|error| in_frame(args, count, error))?;
                let squared = [
                    squared_difference(&decoded.planes[0].to_unorm8(), &picture[0]),
                    squared_difference(&decoded.planes[1].to_unorm8(), &picture[1]),
                    squared_difference(&decoded.planes[2].to_unorm8(), &picture[2]),
                ];
                let samples = [
                    picture[0].len() as u64,
                    picture[1].len() as u64,
                    picture[2].len() as u64,
                ];
                (error, squared, samples)
            }
            y4m::BitDepth::Ten => {
                let src_u10 = [
                    y4m::decode_u10_le_plane(&picture[0])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                    y4m::decode_u10_le_plane(&picture[1])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                    y4m::decode_u10_le_plane(&picture[2])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                ];
                let src_unorm16 = [
                    y4m::u10_plane_to_unorm16(&picture[0])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                    y4m::u10_plane_to_unorm16(&picture[1])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                    y4m::u10_plane_to_unorm16(&picture[2])
                        .map_err(|error| format!("{}: {error}", args.source.display()))?,
                ];
                let planes = [
                    &src_unorm16[0][..],
                    &src_unorm16[1][..],
                    &src_unorm16[2][..],
                ];
                let error = coefficient_error_u16(planes, &view)
                    .map_err(|error| in_frame(args, count, error))?;
                let decoded = decode_frame(&view).map_err(|error| in_frame(args, count, error))?;
                let dec_u10 = [
                    decoded.planes[0]
                        .to_unorm16()
                        .into_iter()
                        .map(y4m::unorm16_to_u10)
                        .collect::<Vec<_>>(),
                    decoded.planes[1]
                        .to_unorm16()
                        .into_iter()
                        .map(y4m::unorm16_to_u10)
                        .collect::<Vec<_>>(),
                    decoded.planes[2]
                        .to_unorm16()
                        .into_iter()
                        .map(y4m::unorm16_to_u10)
                        .collect::<Vec<_>>(),
                ];
                let squared = [
                    squared_difference_u16(&dec_u10[0], &src_u10[0]),
                    squared_difference_u16(&dec_u10[1], &src_u10[1]),
                    squared_difference_u16(&dec_u10[2], &src_u10[2]),
                ];
                let samples = [
                    src_u10[0].len() as u64,
                    src_u10[1].len() as u64,
                    src_u10[2].len() as u64,
                ];
                (error, squared, samples)
            }
        };

        let peak = match header.bit_depth {
            y4m::BitDepth::Eight => 255.0,
            y4m::BitDepth::Ten => 1023.0,
        };
        let mut line = format!("frame {count:>4}: {:>9} bytes", frame.len());
        for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let squared = decoded_squared[i];
            let samples = decoded_samples[i];
            squared_total[i] += squared;
            samples_total[i] += samples;
            line.push_str(&format!(
                "  {name} {} dB",
                format_db(psnr(squared, samples, peak))
            ));
        }
        line.push_str(&format!("  {}", describe(&error)));
        out!("{line}");

        bytes_total += frame.len();
        for i in 0..3 {
            error_total.weighted[i] += error.weighted[i];
            error_total.squared[i] += error.squared[i];
        }
        count += 1;
    }
    if count == 0 {
        return Err("there are no frames to measure".into());
    }

    let peak = match header.bit_depth {
        y4m::BitDepth::Eight => 255.0,
        y4m::BitDepth::Ten => 1023.0,
    };
    // The PSNR of all frames together, from their mean squared difference,
    // which a frame without any keeps finite.
    let frames = count as f64;
    let mut line = format!("mean:       {:>9} bytes", bytes_total / count);
    for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
        let db = psnr(squared_total[i], samples_total[i], peak);
        line.push_str(&format!("  {name} {} dB", format_db(db)));
    }
    let mean = CoefficientError {
        weighted: error_total.weighted.map(|e| e / frames),
        squared: error_total.squared.map(|e| e / frames),
    };
    line.push_str(&format!("  {}", describe(&mean)));
    out!("{line}");
    Ok(())
}

/// The sum of the squared differences of two 8-bit planes.
fn squared_difference(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)).pow(2))
        .sum()
}

/// The sum of the squared differences of two 16-bit planes.
fn squared_difference_u16(a: &[u16], b: &[u16]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)).pow(2))
        .sum()
}

/// PSNR in dB of `squared` over `samples` with peak sample `peak`; infinite for none.
fn psnr(squared: u64, samples: u64, peak: f64) -> f64 {
    if squared == 0 {
        return f64::INFINITY;
    }
    10.0 * (peak * peak / (squared as f64 / samples as f64)).log10()
}

/// The weighted coefficient error, in total and per component.
fn describe(error: &CoefficientError) -> String {
    let [y, cb, cr] = error.weighted;
    format!(
        "weighted error {:.4e} (Y {y:.3e}, Cb {cb:.3e}, Cr {cr:.3e})",
        error.weighted_total()
    )
}

/// `error` with the stream it came from.
fn in_stream(args: &Args, error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    format!("{}: {error}", args.stream.display()).into()
}

/// `error` with the stream and the frame it came from.
fn in_frame(
    args: &Args,
    frame: usize,
    error: impl std::fmt::Display,
) -> Box<dyn std::error::Error> {
    format!("{}: frame {frame}: {error}", args.stream.display()).into()
}
