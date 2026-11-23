//! `emberwave-cli bench`: measure the decode time per frame.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::ValueEnum;
use emberwave_core::bitstream::{Component, Pwv1File};
use emberwave_core::gpu::{
    Decoder, PLANE_FORMAT, PLANE_FORMAT_R16, TimedOutput, TimedUpload, TimingOptions, Timings,
};

use crate::stats::Summary;
use crate::{ColorArgs, GpuArgs, Result, chroma_name};

/// What the benchmark decodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Output {
    /// Y, Cb and Cr planes (8-bit) in packed GPU buffer memory
    Yuv,
    /// RGB in GPU memory: also runs the colour conversion
    Rgb,
    /// Y, Cb and Cr 8-bit plane textures (`R8Unorm`)
    R8,
    /// Y, Cb and Cr 16-bit normalised plane textures (`R16Unorm`)
    R16,
}

/// How the benchmark uploads frame data to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum UploadMode {
    /// Persistent mapped upload buffer (or shared-memory in-place write)
    Mapped,
    /// `Queue::write_buffer`, matching the FFI `submit_*` paths
    Queue,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The file to decode; its frames are decoded in turn, as often as needed
    input: PathBuf,

    /// Number of frames to measure (default: one pass over the file)
    #[arg(long)]
    frames: Option<usize>,

    /// Frames decoded first and not counted, at least
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Seconds of decoding before the counted frames, at least: a GPU runs
    /// slower for the first moments of a load
    #[arg(long, default_value_t = 1.0, value_parser = parse_seconds)]
    warmup_seconds: f64,

    /// What to decode to
    #[arg(long, value_enum, default_value_t = Output::Yuv)]
    output: Output,

    /// How frame data is uploaded to the GPU
    #[arg(long, value_enum, default_value_t = UploadMode::Mapped)]
    upload: UploadMode,

    /// Attach QueueTurns to the GPU context to measure queue-turn overhead
    #[arg(long)]
    queue_turns: bool,

    /// Split the inverse DWT into five compute passes so each level is timed
    #[arg(long)]
    per_level: bool,

    /// Also copy the picture back to the CPU, and time all of it
    #[arg(long)]
    readback: bool,

    /// Print the result as JSON
    #[arg(long)]
    json: bool,

    #[command(flatten)]
    gpu: GpuArgs,

    #[command(flatten)]
    color: ColorArgs,
}

/// Parses a number of seconds: finite, and between 0 and an hour.
pub(crate) fn parse_seconds(text: &str) -> std::result::Result<f64, String> {
    let seconds: f64 = text.parse().map_err(|error| format!("{error}"))?;
    if (0.0..=3600.0).contains(&seconds) {
        Ok(seconds)
    } else {
        Err("must be between 0 and 3600".to_owned())
    }
}

/// A named series of timings in milliseconds.
pub(crate) struct Series {
    pub name: &'static str,
    pub samples: Vec<f64>,
}

impl Series {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            samples: Vec::new(),
        }
    }
}

pub(crate) fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Runs `step` with 0, 1, 2 and so on for at least `frames` frames and at
/// least `seconds` seconds, and returns how many it ran: a GPU that has been
/// idle clocks down, and runs the first moments of a load noticeably slower.
pub(crate) fn warm_up(
    frames: usize,
    seconds: f64,
    mut step: impl FnMut(usize) -> Result<()>,
) -> Result<usize> {
    let time = Duration::from_secs_f64(seconds);
    let start = Instant::now();
    let mut done = 0;
    while done < frames || start.elapsed() < time {
        step(done)?;
        done += 1;
    }
    Ok(done)
}

/// Prints mean, median, 95th and 99th percentiles, minimum and maximum of each series.
pub(crate) fn print_table(series: &[&Series]) -> Result<()> {
    out!(
        "{:<20}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}   (ms per frame)",
        "",
        "mean",
        "p50",
        "p95",
        "p99",
        "min",
        "max"
    );
    for s in series {
        if let Some(summary) = Summary::of(&s.samples) {
            out!(
                "{:<20}{:>9.3}{:>9.3}{:>9.3}{:>9.3}{:>9.3}{:>9.3}",
                s.name,
                summary.mean,
                summary.p50,
                summary.p95,
                summary.p99,
                summary.min,
                summary.max
            );
        }
    }
    Ok(())
}

/// A JSON object of `fields`, whose values are JSON already, then the unit
/// and the summary of each series. Built by hand: the shape is small and flat.
pub(crate) fn json_object(fields: &[(&str, String)], series: &[&Series]) -> String {
    let mut out = String::from("{");
    for (name, value) in fields {
        let _ = write!(out, "{}:{value},", quote(name));
    }
    out.push_str("\"unit\":\"ms\",\"series\":{");
    for (i, s) in series.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if let Some(summary) = Summary::of(&s.samples) {
            let _ = write!(
                out,
                "{}:{{\"mean\":{:.6},\"p50\":{:.6},\"p95\":{:.6},\"p99\":{:.6},\"min\":{:.6},\"max\":{:.6}}}",
                quote(s.name),
                summary.mean,
                summary.p50,
                summary.p95,
                summary.p99,
                summary.min,
                summary.max
            );
        }
    }
    out.push_str("}}");
    out
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == Some(0) {
        return Err("--frames must be at least 1".into());
    }
    if args.readback && matches!(args.output, Output::R8 | Output::R16) {
        return Err("--readback is only supported with --output yuv or --output rgb".into());
    }

    let bytes = crate::read_file(&args.input)?;
    let file = Pwv1File::parse(&bytes)?;
    let stream = file.header().stream;
    let frames: Vec<&[u8]> = file.frames().collect::<std::result::Result<_, _>>()?;
    if frames.is_empty() {
        return Err("the file has no frames".into());
    }

    let context = args.gpu.create_context_with_turns(true, args.queue_turns)?;
    let adapter = context.adapter_info();
    let mut decoder = Decoder::new(&context, stream.width, stream.height, stream.chroma)?;
    let color = args.color.resolve(&stream);

    let plane_textures = match args.output {
        Output::R8 => Some(
            Component::ALL
                .map(|component| decoder.create_plane_texture_with_format(component, PLANE_FORMAT)),
        ),
        Output::R16 => Some(Component::ALL.map(|component| {
            decoder.create_plane_texture_with_format(component, PLANE_FORMAT_R16)
        })),
        Output::Yuv | Output::Rgb => None,
    }
    .map(|results| {
        let [y, cb, cr] = results;
        Ok::<_, emberwave_core::gpu::GpuError>([y?, cb?, cr?])
    })
    .transpose()?;
    let plane_views = plane_textures.as_ref().map(|textures| {
        textures
            .each_ref()
            .map(|t| t.create_view(&Default::default()))
    });

    let timing_options = {
        let mut opts = TimingOptions::default();
        opts.per_level_idwt = args.per_level;
        opts.upload = match args.upload {
            UploadMode::Mapped => TimedUpload::Mapped,
            UploadMode::Queue => TimedUpload::QueueWrite,
        };
        opts
    };

    let count = args.frames.unwrap_or(frames.len());

    // Decodes frame `i` (the file's frames in turn) and returns the timings.
    let mut decode = |i: usize| -> Result<Timings> {
        let frame = frames[i % frames.len()];
        if args.readback {
            let start = Instant::now();
            match args.output {
                Output::Yuv => {
                    decoder.decode(frame)?;
                }
                Output::Rgb => {
                    decoder.decode_rgb(frame, &color)?;
                }
                Output::R8 | Output::R16 => unreachable!("checked above"),
            }
            Ok({
                let mut timings = Timings::default();
                timings.prepare = start.elapsed();
                timings.gpu_wall = Duration::ZERO;
                timings.stages = None;
                timings
            })
        } else if let Some(views) = plane_views.as_ref() {
            Ok(decoder.time_plane_textures(frame, views.each_ref(), timing_options)?)
        } else {
            let output = match args.output {
                Output::Yuv => TimedOutput::Yuv,
                Output::Rgb => TimedOutput::Rgb(color),
                Output::R8 | Output::R16 => unreachable!("handled by plane_views"),
            };
            Ok(decoder.time_frame_with(frame, output, timing_options)?)
        }
    };

    let warmup = warm_up(args.warmup, args.warmup_seconds, |i| decode(i).map(drop))?;

    let mut total = Series::new("total");
    let mut assemble = Series::new("cpu assemble");
    let mut upload = Series::new("cpu upload");
    let mut queue_turn = Series::new("queue turn");
    let mut submit = Series::new("cpu submit");
    let mut prepare = Series::new("cpu prepare");
    let mut gpu_wall = Series::new("gpu wall");
    let mut dequant = Series::new("dequant");
    let mut idwt = Series::new("idwt");
    let mut idwt_levels = [
        Series::new("idwt L4"),
        Series::new("idwt L3"),
        Series::new("idwt L2"),
        Series::new("idwt L1"),
        Series::new("idwt L0"),
    ];
    let mut yuv2rgb = Series::new("yuv2rgb");
    let mut gpu_sum = Series::new("gpu (timestamps)");

    for i in 0..count {
        let timings = decode(i)?;
        if args.readback {
            total.samples.push(ms(timings.prepare));
            continue;
        }

        total.samples.push(ms(timings.prepare + timings.gpu_wall));
        assemble.samples.push(ms(timings.assemble));
        upload.samples.push(ms(timings.upload));
        if args.queue_turns {
            queue_turn.samples.push(ms(timings.queue_turn));
        }
        submit.samples.push(ms(timings.submit));
        prepare.samples.push(ms(timings.prepare));
        gpu_wall.samples.push(ms(timings.gpu_wall));
        if let Some(stages) = timings.stages {
            dequant.samples.push(ms(stages.dequant));
            idwt.samples.push(ms(stages.idwt));
            if let Some(levels) = stages.idwt_levels {
                for (series, dur) in idwt_levels.iter_mut().zip(levels) {
                    series.samples.push(ms(dur));
                }
            }
            if let Some(duration) = stages.yuv2rgb {
                yuv2rgb.samples.push(ms(duration));
            }
            gpu_sum.samples.push(ms(stages.total()));
        }
    }

    let series: Vec<&Series> = [
        &total,
        &prepare,
        &assemble,
        &upload,
        &queue_turn,
        &submit,
        &gpu_wall,
        &gpu_sum,
        &dequant,
        &idwt,
        &idwt_levels[0],
        &idwt_levels[1],
        &idwt_levels[2],
        &idwt_levels[3],
        &idwt_levels[4],
        &yuv2rgb,
    ]
    .into_iter()
    .filter(|s| !s.samples.is_empty())
    .collect();

    let description = format!(
        "{}x{} {}, decoded to {}{}",
        stream.width,
        stream.height,
        chroma_name(stream.chroma),
        match args.output {
            Output::Yuv => "YUV",
            Output::Rgb => "RGB",
            Output::R8 => {
                if decoder.stores_planes() {
                    "R8Unorm plane textures (direct store)"
                } else {
                    "R8Unorm plane textures (packed+copy)"
                }
            }
            Output::R16 => "R16Unorm plane textures (direct store)",
        },
        if args.readback {
            " and read back to the CPU"
        } else {
            " in GPU memory"
        },
    );

    if args.json {
        out!(
            "{}",
            json(
                args,
                &adapter.name,
                &format!("{:?}", adapter.backend),
                &description,
                (count, warmup),
                &decoder,
                &series
            )
        );
    } else {
        out!("adapter:  {} ({:?})", adapter.name, adapter.backend);
        out!("decoding: {description}");
        out!(
            "mode:     upload={:?}, queue_turns={}, per_level={}, shared_uploads={}, r16_planes={}",
            args.upload,
            args.queue_turns,
            args.per_level,
            context.shares_uploads(),
            decoder.supports_plane_format(PLANE_FORMAT_R16),
        );
        out!("frames:   {count} measured after {warmup} warm-up frames");
        out!();
        print_table(&series)?;
    }
    Ok(())
}

/// The result as JSON. `frames` is the number of measured frames and of
/// warm-up frames that were decoded.
fn json(
    args: &Args,
    adapter: &str,
    backend: &str,
    description: &str,
    (frames, warmup): (usize, usize),
    decoder: &Decoder,
    series: &[&Series],
) -> String {
    let upload_mode = match args.upload {
        UploadMode::Mapped => "mapped",
        UploadMode::Queue => "queue",
    };
    json_object(
        &[
            ("input", quote(&args.input.display().to_string())),
            ("adapter", quote(adapter)),
            ("backend", quote(backend)),
            ("description", quote(description)),
            ("frames", frames.to_string()),
            ("warmup", warmup.to_string()),
            ("readback", args.readback.to_string()),
            ("upload_mode", quote(upload_mode)),
            ("queue_turns", args.queue_turns.to_string()),
            ("per_level", args.per_level.to_string()),
            ("stores_planes", decoder.stores_planes().to_string()),
            (
                "r16_planes",
                decoder.supports_plane_format(PLANE_FORMAT_R16).to_string(),
            ),
        ],
        series,
    )
}

/// A JSON string literal.
pub(crate) fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_warm_up_time_must_be_a_sensible_number_of_seconds() {
        assert_eq!(parse_seconds("0"), Ok(0.0));
        assert_eq!(parse_seconds("1.5"), Ok(1.5));
        for bad in ["-1", "nan", "inf", "3601", "soon"] {
            assert!(parse_seconds(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(quote("plain"), "\"plain\"");
        assert_eq!(quote("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(quote("line\nbreak\ttab"), "\"line\\nbreak\\ttab\"");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
        assert_eq!(quote("한국어"), "\"한국어\"");
    }
}
