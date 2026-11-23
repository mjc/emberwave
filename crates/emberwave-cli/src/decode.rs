//! `emberwave-cli decode`: decode a file to PNG images or a Y4M video.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::ValueEnum;
use emberwave_core::bitstream::{ChromaSiting, ChromaSubsampling, Component, Pwv1File, YcbcrRange};
use emberwave_core::gpu::{Decoder, GpuContext, GpuError, PLANE_FORMAT_R16};

use crate::stats::Summary;
use crate::{ColorArgs, GpuArgs, MatrixChoice, RangeChoice, Result, read_file, y4m};

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// One RGB PNG per frame
    Png,
    /// One Y4M file with the YUV planes of all frames, without colour conversion
    Y4m,
}

/// Bit depth for `--format y4m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BitDepthChoice {
    /// Take the stream's reference bit depth (10-bit for 10, 8-bit otherwise)
    Auto,
    /// 8-bit planar Y4M (`C420jpeg`, `C420mpeg2`, or `C444`)
    #[value(name = "8")]
    Eight,
    /// 10-bit planar Y4M (`C420p10` or `C444p10`)
    #[value(name = "10")]
    Ten,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The file to decode
    input: PathBuf,

    /// Directory for the output; created if it does not exist
    #[arg(long)]
    out_dir: PathBuf,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Png)]
    format: Format,

    /// Sample bit depth when decoding to --format y4m
    #[arg(long, value_enum, default_value_t = BitDepthChoice::Auto)]
    bit_depth: BitDepthChoice,

    /// Decode at most this many frames
    #[arg(long)]
    frames: Option<usize>,

    #[command(flatten)]
    gpu: GpuArgs,

    #[command(flatten)]
    color: ColorArgs,
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == Some(0) {
        return Err("--frames must be at least 1".into());
    }
    if args.format == Format::Png && args.bit_depth != BitDepthChoice::Auto {
        return Err("--bit-depth applies only to --format y4m; PNG output is 8-bit RGB".into());
    }
    // The conversion options do nothing to planes that are written as they
    // are; accepting them would let the file say something the data does not.
    if args.format == Format::Y4m
        && (args.color.matrix != MatrixChoice::Auto || args.color.range != RangeChoice::Auto)
    {
        return Err(
            "--matrix and --range set the RGB conversion; Y4M output keeps the planes as decoded"
                .into(),
        );
    }

    let bytes = read_file(&args.input)?;
    let file = Pwv1File::parse(&bytes)?;
    let header = *file.header();
    let stream = header.stream;
    // Before anything is written, as bench and compare refuse it too.
    if file.frames().next().is_none() {
        return Err("the file has no frames".into());
    }

    let y4m_bit_depth = match args.bit_depth {
        BitDepthChoice::Auto => {
            if header.reference_bit_depth == 10 {
                y4m::BitDepth::Ten
            } else {
                y4m::BitDepth::Eight
            }
        }
        BitDepthChoice::Eight => y4m::BitDepth::Eight,
        BitDepthChoice::Ten => y4m::BitDepth::Ten,
    };

    // Before the GPU: a directory that cannot be made fails at once.
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|error| format!("{}: {error}", args.out_dir.display()))?;
    let context = args.gpu.create_context(false)?;
    let mut decoder = Decoder::new(&context, stream.width, stream.height, stream.chroma)?;
    let color = args.color.resolve(&stream);

    if args.format == Format::Y4m
        && y4m_bit_depth == y4m::BitDepth::Ten
        && !decoder.supports_plane_format(PLANE_FORMAT_R16)
    {
        return Err(
            "10-bit Y4M decode requires 16-bit normalised plane textures \
             (TEXTURE_FORMAT_16BIT_NORM), which this GPU adapter does not support"
                .into(),
        );
    }

    let stem = args
        .input
        .file_stem()
        .map_or_else(|| "frame".to_owned(), |s| s.to_string_lossy().into_owned());

    let use_r16_readback = args.format == Format::Y4m
        && (y4m_bit_depth == y4m::BitDepth::Ten
            || (header.reference_bit_depth == 10
                && stream.ycbcr_range == YcbcrRange::Limited
                && decoder.supports_plane_format(PLANE_FORMAT_R16)));
    let mut r16_readback = if use_r16_readback {
        Some(R16PlaneReadback::new(
            &context,
            &decoder,
            y4m_bit_depth,
            header.reference_bit_depth,
            stream.ycbcr_range,
        )?)
    } else {
        None
    };

    let mut video = match args.format {
        Format::Y4m => {
            let info = y4m::StreamInfo {
                width: stream.width,
                height: stream.height,
                chroma: stream.chroma,
                siting: stream.chroma_siting,
                frame_rate: (header.frame_rate_num, header.frame_rate_den),
                range: stream.ycbcr_range,
                bit_depth: y4m_bit_depth,
            };
            if !info.has_frame_rate() {
                let (num, den) = y4m::DEFAULT_FRAME_RATE;
                err!("warning: the file has no frame rate, the Y4M file says {num}:{den}");
            }
            if info.bit_depth == y4m::BitDepth::Ten
                && info.chroma == ChromaSubsampling::Yuv420
                && info.siting == ChromaSiting::Left
            {
                err!(
                    "warning: Y4M C420p10 cannot signal left chroma siting; writing C420p10 (emberwave reads C420p10 as centred)"
                );
            }
            Some(y4m::Writer::create(
                &args.out_dir.join(format!("{stem}.y4m")),
                &info,
            )?)
        }
        Format::Png => None,
    };

    let limit = args.frames.unwrap_or(usize::MAX);
    let mut times_ms = Vec::new();
    for (index, frame) in file.frames().enumerate().take(limit) {
        let frame = frame?;

        // The time covers everything up to having the picture on the CPU.
        let start = Instant::now();
        match args.format {
            Format::Png => {
                let rgb = decoder.decode_rgb(frame, &color)?;
                let elapsed = start.elapsed();
                let path = args.out_dir.join(format!("{stem}_{index:06}.png"));
                write_png(&path, rgb.width, rgb.height, &rgb.data)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                times_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
            Format::Y4m => {
                if let Some(readback) = r16_readback.as_mut() {
                    readback.decode_to_planes(&context, &mut decoder, frame)?;
                    times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    if let Some(video) = video.as_mut() {
                        let [y, cb, cr] = &readback.out_planes;
                        video.write_frame([y, cb, cr])?;
                    }
                } else {
                    let mut yuv = decoder.decode(frame)?;
                    if y4m_bit_depth == y4m::BitDepth::Eight
                        && header.reference_bit_depth == 10
                        && stream.ycbcr_range == YcbcrRange::Limited
                    {
                        for plane in &mut yuv.planes {
                            for sample in &mut plane.data {
                                *sample = y4m::unorm8_to_u8_limited_from_u10(*sample);
                            }
                        }
                    }
                    times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    if let Some(video) = video.as_mut() {
                        let [y, cb, cr] = &yuv.planes;
                        video.write_frame([&y.data, &cb.data, &cr.data])?;
                    }
                }
            }
        }
        err!("frame {index:06}: {:.3} ms", times_ms[index]);
    }

    if let Some(video) = video {
        video.finish()?;
    }

    match Summary::of(&times_ms) {
        Some(summary) => out!(
            "decoded {} frames to {}: mean {:.3} ms, p50 {:.3} ms, p99 {:.3} ms per frame",
            times_ms.len(),
            args.out_dir.display(),
            summary.mean,
            summary.p50,
            summary.p99
        ),
        None => return Err("the file has no frames".into()),
    }
    Ok(())
}

fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    // Speed over size: these images are for looking at and for tests.
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgb)?;
    // Writes the end of the file and flushes it, and says if that failed,
    // which dropping the writer would not.
    writer.finish()?;
    Ok(())
}

/// Runs a GPU API closure inside `wgpu` validation, internal, and OOM error
/// scopes so failures become `Err` values instead of reaching the default panic
/// handler (decision D22).
fn with_error_scope<T>(device: &wgpu::Device, body: impl FnOnce() -> T) -> Result<T> {
    let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = body();
    let validation = pollster::block_on(validation_scope.pop());
    let internal = pollster::block_on(internal_scope.pop());
    let memory = pollster::block_on(memory_scope.pop());

    if let Some(error) = validation.or(internal).or(memory) {
        Err(GpuError::Device(error.to_string()).into())
    } else {
        Ok(value)
    }
}

/// Reusable `R16Unorm` plane textures and staging buffers for decoding 10-bit
/// Y4M frames (or 10-to-8 limited-range downsampled Y4M frames) from the GPU.
struct R16PlaneReadback {
    textures: [wgpu::Texture; 3],
    views: [wgpu::TextureView; 3],
    buffers: [wgpu::Buffer; 3],
    row_bytes: [u32; 3],
    padded_bytes_per_row: [u32; 3],
    heights: [u32; 3],
    target_bit_depth: y4m::BitDepth,
    source_bit_depth: u8,
    range: YcbcrRange,
    out_planes: [Vec<u8>; 3],
}

impl R16PlaneReadback {
    fn new(
        context: &GpuContext,
        decoder: &Decoder,
        target_bit_depth: y4m::BitDepth,
        source_bit_depth: u8,
        range: YcbcrRange,
    ) -> Result<Self> {
        if let Some(reason) = context.lost() {
            return Err(GpuError::DeviceLost(reason).into());
        }
        let layout = decoder.layout();
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut textures = Vec::with_capacity(3);
        let mut views = Vec::with_capacity(3);
        let mut buffers = Vec::with_capacity(3);
        let mut row_bytes = [0u32; 3];
        let mut padded_bytes_per_row = [0u32; 3];
        let mut heights = [0u32; 3];
        let mut out_planes: [Vec<u8>; 3] = Default::default();

        for (i, component) in Component::ALL.into_iter().enumerate() {
            let texture = decoder.create_plane_texture_with_format(component, PLANE_FORMAT_R16)?;
            let (width, height) = layout.plane_size(component);
            let row = width * 2;
            let padded = row.div_ceil(align) * align;
            let (view, buffer) = with_error_scope(context.device(), || {
                let view = texture.create_view(&Default::default());
                let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("r16 y4m readback"),
                    size: u64::from(padded) * u64::from(height),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (view, buffer)
            })?;
            textures.push(texture);
            views.push(view);
            buffers.push(buffer);
            row_bytes[i] = row;
            padded_bytes_per_row[i] = padded;
            heights[i] = height;
            let out_row = width as usize * target_bit_depth.bytes_per_sample();
            out_planes[i] = vec![0u8; out_row * height as usize];
        }
        if let Some(reason) = context.lost() {
            return Err(GpuError::DeviceLost(reason).into());
        }

        Ok(Self {
            textures: textures
                .try_into()
                .map_err(|_| "expected three R16 textures")?,
            views: views.try_into().map_err(|_| "expected three R16 views")?,
            buffers: buffers
                .try_into()
                .map_err(|_| "expected three R16 readback buffers")?,
            row_bytes,
            padded_bytes_per_row,
            heights,
            target_bit_depth,
            source_bit_depth,
            range,
            out_planes,
        })
    }

    fn decode_to_planes(
        &mut self,
        context: &GpuContext,
        decoder: &mut Decoder,
        frame: &[u8],
    ) -> Result<()> {
        if let Some(reason) = context.lost() {
            return Err(GpuError::DeviceLost(reason).into());
        }
        decoder.clear();
        decoder.push_packet(frame)?;
        if !decoder.is_ready(false) {
            return Err(GpuError::IncompleteFrame {
                received: decoder.received_blocks(),
                total: decoder.total_blocks(),
            }
            .into());
        }
        decoder.decode_assembled_to_plane_textures(self.views.each_ref())?;

        with_error_scope(context.device(), || {
            let mut encoder =
                context
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("r16 y4m readback"),
                    });
            for i in 0..3 {
                encoder.copy_texture_to_buffer(
                    self.textures[i].as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.buffers[i],
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(self.padded_bytes_per_row[i]),
                            rows_per_image: Some(self.heights[i]),
                        },
                    },
                    self.textures[i].size(),
                );
            }
            context.queue().submit([encoder.finish()]);
        })?;

        // Unmap only buffers whose `map_async` succeeded so an error never
        // calls `unmap()` on an idle buffer, and always run `unmap()` inside a
        // wgpu error scope (decision D22).
        struct MappedGuard<'a> {
            device: &'a wgpu::Device,
            buffers: &'a [wgpu::Buffer; 3],
            mapped: [bool; 3],
        }
        impl MappedGuard<'_> {
            fn unmap_all(&mut self) -> Result<()> {
                if !self.mapped.iter().any(|&m| m) {
                    return Ok(());
                }
                let buffers = self.buffers;
                let mapped = std::mem::take(&mut self.mapped);
                with_error_scope(self.device, || {
                    for (buffer, is_mapped) in buffers.iter().zip(mapped) {
                        if is_mapped {
                            buffer.unmap();
                        }
                    }
                })
            }
        }
        impl Drop for MappedGuard<'_> {
            fn drop(&mut self) {
                let _ = self.unmap_all();
            }
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        with_error_scope(context.device(), || {
            for (i, buffer) in self.buffers.iter().enumerate() {
                let sender = sender.clone();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    let _ = sender.send((i, res));
                });
            }
        })?;
        drop(sender);

        let mut guard = MappedGuard {
            device: context.device(),
            buffers: &self.buffers,
            mapped: [false; 3],
        };
        if let Err(error) = context.device().poll(wgpu::PollType::wait_indefinitely()) {
            while let Ok((i, res)) = receiver.try_recv() {
                if res.is_ok() {
                    guard.mapped[i] = true;
                }
            }
            return Err(format!("waiting for 16-bit plane readback failed: {error}").into());
        }
        let mut map_error = None;
        for _ in 0..3 {
            let (i, res) = receiver
                .recv()
                .map_err(|error| format!("16-bit plane readback failed: {error}"))?;
            match res {
                Ok(()) => guard.mapped[i] = true,
                Err(error) => {
                    if map_error.is_none() {
                        map_error = Some(error);
                    }
                }
            }
        }
        if let Some(reason) = context.lost() {
            return Err(GpuError::DeviceLost(reason).into());
        }
        if let Some(error) = map_error {
            return Err(format!("mapping 16-bit plane readback buffer failed: {error}").into());
        }

        for i in 0..3 {
            let mapped = self.buffers[i]
                .slice(..)
                .get_mapped_range()
                .map_err(|error| format!("reading 16-bit plane readback buffer failed: {error}"))?;
            let row_bytes = self.row_bytes[i] as usize;
            let padded = self.padded_bytes_per_row[i] as usize;
            let out = &mut self.out_planes[i];
            match self.target_bit_depth {
                y4m::BitDepth::Ten => {
                    for (dst_row, src_row) in out
                        .chunks_exact_mut(row_bytes)
                        .zip(mapped.chunks_exact(padded).take(self.heights[i] as usize))
                    {
                        for (dst, src) in dst_row
                            .chunks_exact_mut(2)
                            .zip(src_row[..row_bytes].chunks_exact(2))
                        {
                            let unorm16 = u16::from_le_bytes([src[0], src[1]]);
                            let u10 = y4m::unorm16_to_u10_with_range(
                                unorm16,
                                self.source_bit_depth,
                                self.range,
                            );
                            dst.copy_from_slice(&u10.to_le_bytes());
                        }
                    }
                }
                y4m::BitDepth::Eight => {
                    let out_row_bytes = row_bytes / 2;
                    for (dst_row, src_row) in out
                        .chunks_exact_mut(out_row_bytes)
                        .zip(mapped.chunks_exact(padded).take(self.heights[i] as usize))
                    {
                        for (dst, src) in
                            dst_row.iter_mut().zip(src_row[..row_bytes].chunks_exact(2))
                        {
                            let unorm16 = u16::from_le_bytes([src[0], src[1]]);
                            *dst = y4m::unorm16_to_u8_limited_from_u10(unorm16);
                        }
                    }
                }
            }
        }
        guard.unmap_all()?;
        Ok(())
    }
}
