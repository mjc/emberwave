//! `emberwave-cli gen-stream`: generate deterministic synthetic `.pwv1` test
//! streams for repeatable decode benchmarks across platforms.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::ValueEnum;
use emberwave_core::bitstream::consts::HEADER_BYTES;
use emberwave_core::bitstream::{
    ChromaSiting, ChromaSubsampling, ColorPrimaries, Pwv1Header, SequenceHeader, TransferFunction,
    YcbcrRange, YcbcrTransform,
};
use emberwave_core::encode::{self, Signalling};
use emberwave_core::gpu;

use crate::encode::SignalledMatrix;
use crate::{Backend, GpuArgs, Result, chroma_name};

/// Synthetic picture pattern for stream generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Pattern {
    /// Smooth mostly-flat desktop/HUD frame with sparse detail (high compression)
    Idle,
    /// Colour bars, sky gradient, moving shaded disc, diagonal grid, subtle grain
    Scene,
    /// Fast-moving multi-disc scene with high-contrast texture across the whole frame
    HighMotion,
    /// Uniform pseudo-random noise (worst-case coefficient density)
    Noise,
}

/// Chroma subsampling choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChromaChoice {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "444")]
    Yuv444,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Output PWV1 file path
    #[arg(long)]
    out: PathBuf,

    /// Picture width in pixels
    #[arg(long, default_value_t = 3840)]
    width: u32,

    /// Picture height in pixels
    #[arg(long, default_value_t = 2160)]
    height: u32,

    /// Chroma subsampling
    #[arg(long, value_enum, default_value_t = ChromaChoice::Yuv420)]
    chroma: ChromaChoice,

    /// Synthetic picture pattern
    #[arg(long, value_enum, default_value_t = Pattern::Scene)]
    pattern: Pattern,

    /// Number of frames to generate
    #[arg(long, default_value_t = 60)]
    frames: usize,

    /// Frame rate in frames per second (numerator)
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Target bitrate in Mbit/s at `--fps` (mutually exclusive with `--bytes-per-frame`)
    #[arg(long, conflicts_with = "bytes_per_frame")]
    bitrate_mbps: Option<f64>,

    /// Maximum bytes per frame (mutually exclusive with `--bitrate-mbps`)
    #[arg(long)]
    bytes_per_frame: Option<usize>,

    /// Colour signalling to record in the stream
    #[arg(long, value_enum, default_value_t = SignalledMatrix::Bt2020)]
    matrix: SignalledMatrix,

    /// Reference bit depth recorded in the PWV1 file header (8 or 10)
    #[arg(long, default_value_t = 10)]
    bit_depth: u8,

    /// Encode on the GPU instead of the CPU
    #[arg(long)]
    gpu: bool,

    #[command(flatten)]
    gpu_args: GpuArgs,
}

enum Encoder {
    Cpu(Box<encode::Encoder>),
    Gpu(Box<gpu::Encoder>),
}

impl Encoder {
    fn stream(&self) -> SequenceHeader {
        match self {
            Self::Cpu(encoder) => encoder.stream(),
            Self::Gpu(encoder) => encoder.stream(),
        }
    }

    fn encode(&mut self, planes: [&[u8]; 3], budget: usize) -> Result<Vec<u8>> {
        Ok(match self {
            Self::Cpu(encoder) => encoder.encode(planes, budget)?,
            Self::Gpu(encoder) => encoder.encode(planes, budget)?,
        })
    }
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == 0 {
        return Err("--frames must be at least 1".into());
    }
    if args.fps == 0 {
        return Err("--fps must be at least 1".into());
    }
    if args.bit_depth != 8 && args.bit_depth != 10 {
        return Err("--bit-depth must be 8 or 10".into());
    }
    let chroma = match args.chroma {
        ChromaChoice::Yuv420 => ChromaSubsampling::Yuv420,
        ChromaChoice::Yuv444 => ChromaSubsampling::Yuv444,
    };
    if chroma == ChromaSubsampling::Yuv420 && !(args.width | args.height).is_multiple_of(2) {
        return Err("4:2:0 requires even width and height".into());
    }
    if !args.gpu
        && (args.gpu_args.backend != Backend::Auto
            || args.gpu_args.software
            || args.gpu_args.low_power
            || args.gpu_args.adapter.is_some())
    {
        return Err("--backend, --software, --low-power and --adapter require --gpu".into());
    }

    let bytes_per_frame = match (args.bytes_per_frame, args.bitrate_mbps) {
        (Some(bytes), None) => bytes,
        (None, Some(mbps)) => {
            if !mbps.is_finite() || mbps <= 0.0 {
                return Err("--bitrate-mbps must be positive and finite".into());
            }
            ((mbps * 1_000_000.0 / 8.0) / f64::from(args.fps)).round() as usize
        }
        (None, None) => ((150.0 * 1_000_000.0 / 8.0) / f64::from(args.fps)).round() as usize,
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
    };
    if bytes_per_frame < HEADER_BYTES {
        return Err(format!("frame budget {bytes_per_frame} < {HEADER_BYTES}").into());
    }

    let (primaries, transfer, transform) = match args.matrix {
        SignalledMatrix::Bt709 => (
            ColorPrimaries::Srgb,
            TransferFunction::Srgb,
            YcbcrTransform::Bt709,
        ),
        SignalledMatrix::Bt2020 => (
            ColorPrimaries::Bt2020,
            TransferFunction::Pq,
            YcbcrTransform::Bt2020,
        ),
    };
    let mut signalling = Signalling::default();
    signalling.color_primaries = primaries;
    signalling.transfer_function = transfer;
    signalling.ycbcr_transform = transform;
    signalling.ycbcr_range = YcbcrRange::Limited;
    signalling.chroma_siting = ChromaSiting::Center;

    let mut encoder = if args.gpu {
        let context = args.gpu_args.create_context(false)?;
        Encoder::Gpu(Box::new(gpu::Encoder::new(
            &context,
            args.width,
            args.height,
            chroma,
            signalling,
        )?))
    } else {
        Encoder::Cpu(Box::new(encode::Encoder::new(
            args.width,
            args.height,
            chroma,
            signalling,
        )?))
    };

    let file_header = Pwv1Header {
        stream: encoder.stream(),
        frame_rate_num: args.fps,
        frame_rate_den: 1,
        reference_bit_depth: args.bit_depth,
    };

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut partial_name = args.out.file_name().unwrap_or_default().to_os_string();
    partial_name.push(".partial");
    let partial = args.out.with_file_name(partial_name);

    let cw = if chroma == ChromaSubsampling::Yuv444 {
        args.width as usize
    } else {
        (args.width / 2) as usize
    };
    let ch = if chroma == ChromaSubsampling::Yuv444 {
        args.height as usize
    } else {
        (args.height / 2) as usize
    };
    let mut y = vec![0u8; args.width as usize * args.height as usize];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];

    let result = (|| -> Result<usize> {
        let file = File::create(&partial)?;
        let mut out = BufWriter::new(file);
        out.write_all(&file_header.to_bytes())?;
        let mut total_bytes = 0usize;
        for frame_idx in 0..args.frames {
            synthesize_frame(
                args.pattern,
                frame_idx,
                args.width as usize,
                args.height as usize,
                chroma,
                &mut y,
                &mut cb,
                &mut cr,
            );
            let encoded = encoder.encode([&y, &cb, &cr], bytes_per_frame)?;
            out.write_all(&u32::try_from(encoded.len())?.to_le_bytes())?;
            out.write_all(&encoded)?;
            total_bytes += encoded.len();
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        Ok(total_bytes)
    })();

    let total_bytes = match result {
        Ok(total) => total,
        Err(err) => {
            let _ = std::fs::remove_file(&partial);
            return Err(err);
        }
    };
    std::fs::rename(&partial, &args.out)?;

    let mean_bytes = total_bytes / args.frames;
    let actual_mbps = (mean_bytes as f64 * 8.0 * f64::from(args.fps)) / 1_000_000.0;
    out!(
        "generated {} ({}x{} {}, {}-bit, {:?}, {} frames): mean {} bytes/frame ({:.1} Mbps @ {} fps)",
        args.out.display(),
        args.width,
        args.height,
        chroma_name(chroma),
        args.bit_depth,
        args.pattern,
        args.frames,
        mean_bytes,
        actual_mbps,
        args.fps,
    );
    Ok(())
}

fn hash01(x: u32, y: u32, frame: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(frame.wrapping_mul(2_246_822_519))
        .wrapping_add(seed.wrapping_mul(3_266_489_917));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

#[allow(clippy::too_many_arguments)]
fn synthesize_frame(
    pattern: Pattern,
    frame: usize,
    width: usize,
    height: usize,
    chroma: ChromaSubsampling,
    y_plane: &mut [u8],
    cb_plane: &mut [u8],
    cr_plane: &mut [u8],
) {
    let cw = if chroma == ChromaSubsampling::Yuv444 {
        width
    } else {
        width / 2
    };
    let ch = if chroma == ChromaSubsampling::Yuv444 {
        height
    } else {
        height / 2
    };
    let step = if chroma == ChromaSubsampling::Yuv444 {
        1
    } else {
        2
    };

    match pattern {
        Pattern::Idle => {
            // Smooth static backdrop with a small stationary status bar; most
            // high-frequency wavelet blocks stay near zero.
            for py in 0..height {
                let luma = if py < height / 16 {
                    200u8
                } else {
                    (24 + (py * 48) / height.max(1)) as u8
                };
                let row = &mut y_plane[py * width..(py + 1) * width];
                row.fill(luma);
            }
            cb_plane.fill(134);
            cr_plane.fill(124);
        }
        Pattern::Noise => {
            let mut state = 1u32
                .wrapping_mul(2_654_435_761)
                .wrapping_add((frame as u32).wrapping_mul(7919))
                .wrapping_add(1);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            };
            for v in y_plane.iter_mut() {
                *v = next();
            }
            for v in cb_plane.iter_mut() {
                *v = next();
            }
            for v in cr_plane.iter_mut() {
                *v = next();
            }
        }
        Pattern::Scene | Pattern::HighMotion => {
            let aspect = width as f32 / height as f32;
            let motion_scale = if pattern == Pattern::HighMotion {
                0.85
            } else {
                0.15
            };
            let cx = 0.30 + 0.40 * (0.5 + 0.5 * (motion_scale * frame as f32).sin());
            let cy = if pattern == Pattern::HighMotion {
                0.35 + 0.30 * (0.5 + 0.5 * (0.63 * frame as f32).cos())
            } else {
                0.56
            };
            let radius = 0.22f32;
            let shift = if pattern == Pattern::HighMotion {
                frame * 13
            } else {
                0
            };
            const BARS: [[f32; 3]; 8] = [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ];
            let grain_amp = if pattern == Pattern::HighMotion {
                0.06
            } else {
                0.02
            };

            for cy_i in 0..ch {
                let v_base = cy_i * step;
                for cx_i in 0..cw {
                    let mut cb_sum = 0.0f32;
                    let mut cr_sum = 0.0f32;
                    for dy in 0..step {
                        let py = v_base + dy;
                        let v = (py as f32 + 0.5) / height as f32;
                        for dx in 0..step {
                            let px = cx_i * step + dx;
                            let u = (px as f32 + 0.5) / width as f32;
                            let mut c = if v < 0.12 {
                                BARS[((u * 8.0) as usize).min(7)]
                            } else {
                                let t = (v - 0.12) / 0.88;
                                let mut bg = [0.15 + 0.65 * t, 0.35 + 0.25 * t, 0.85 - 0.55 * t];
                                let ddx = (u - cx) * aspect;
                                let ddy = v - cy;
                                let cov = (0.5
                                    + (radius - (ddx * ddx + ddy * ddy).sqrt()) * height as f32)
                                    .clamp(0.0, 1.0);
                                let shade = (0.5 + ddx / (2.0 * radius)).clamp(0.0, 1.0);
                                let disc = [0.95, 0.80 - 0.50 * shade, 0.20 + 0.60 * shade];
                                bg[0] += (disc[0] - bg[0]) * cov;
                                bg[1] += (disc[1] - bg[1]) * cov;
                                bg[2] += (disc[2] - bg[2]) * cov;
                                let grid_active =
                                    pattern == Pattern::HighMotion || (u < 0.35 && v > 0.70);
                                if grid_active
                                    && ((px + py + shift) % 16 < 2
                                        || (px + 65536 - py + shift) % 16 < 2)
                                {
                                    bg = [0.05, 0.05, 0.05];
                                }
                                bg
                            };
                            let grain =
                                (hash01(px as u32, py as u32, frame as u32, 1) - 0.5) * grain_amp;
                            c[0] = (c[0] + grain).clamp(0.0, 1.0);
                            c[1] = (c[1] + grain).clamp(0.0, 1.0);
                            c[2] = (c[2] + grain).clamp(0.0, 1.0);
                            let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                            cb_sum += (c[2] - luma) / 1.8556;
                            cr_sum += (c[0] - luma) / 1.5748;
                            y_plane[py * width + px] =
                                (16.0 + 219.0 * luma).round().clamp(16.0, 235.0) as u8;
                        }
                    }
                    let n = (step * step) as f32;
                    let idx = cy_i * cw + cx_i;
                    cb_plane[idx] = (128.0 + 224.0 * cb_sum / n).round().clamp(16.0, 240.0) as u8;
                    cr_plane[idx] = (128.0 + 224.0 * cr_sum / n).round().clamp(16.0, 240.0) as u8;
                }
            }
        }
    }
}
