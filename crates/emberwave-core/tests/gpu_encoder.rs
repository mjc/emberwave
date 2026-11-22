//! The GPU encoder through its public API: on the sources of the encoder's
//! vectors in `testdata/encoder/`, against the CPU encoder's committed
//! streams; on a device limited to what WebGPU guarantees; frame after frame;
//! from textures as from CPU memory, and from RGB as from the planes it
//! converts to; and on a lost device (decisions D29, D32 and D65).
//!
//! Like `tests/gpu.rs`, they skip without a GPU unless
//! `EMBERWAVE_REQUIRE_GPU` is set.

#![cfg(feature = "gpu")]

mod common;

use common::{
    Y4m, create_context, encoder_dir, encoder_vectors, live_resources, psnr, read_y4m,
    shared_context,
};
use emberwave_core::bitstream::{
    ChromaSubsampling, Component, FrameAssembler, FrameLayout, Header, Pwv1File,
};
use emberwave_core::color::{ColorError, RgbOrder, rgb_to_ycbcr8, yuv_to_rgb8};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{self, EncodeError, Signalling};
use emberwave_core::gpu::{Decoder, Encoder, GpuContext, GpuError, TexturePicture};

/// Skips the current test when there is no GPU.
macro_rules! gpu_or_return {
    () => {
        match shared_context() {
            Some(context) => context,
            None => return,
        }
    };
}

/// A vector of `testdata/encoder/`: its source, budget and committed frames.
struct Vector {
    name: String,
    budget: usize,
    source: Y4m,
    layout: FrameLayout,
    committed: Vec<Vec<u8>>,
}

fn vectors() -> Vec<Vector> {
    encoder_vectors()
        .into_iter()
        .map(|(name, budget)| {
            let source = read_y4m(&encoder_dir().join(format!("{name}.source.y4m")));
            let bytes = std::fs::read(encoder_dir().join(format!("{name}.pwv1"))).expect("stream");
            let file = Pwv1File::parse(&bytes).expect("a PWV1 file");
            let stream = file.header().stream;
            let layout =
                FrameLayout::new(stream.width, stream.height, stream.chroma).expect("layout");
            let committed = file
                .frames()
                .map(|frame| frame.expect("a frame").to_vec())
                .collect();
            Vector {
                name,
                budget,
                source,
                layout,
                committed,
            }
        })
        .collect()
}

fn encoder_for(context: &GpuContext, layout: &FrameLayout) -> Encoder {
    Encoder::new(
        context,
        layout.width(),
        layout.height(),
        layout.chroma(),
        Signalling::default(),
    )
    .expect("an encoder")
}

fn planes(frame: &[Vec<u8>; 3]) -> [&[u8]; 3] {
    [&frame[0], &frame[1], &frame[2]]
}

/// A texture of `format` and `usage` on `context`'s device, filled with
/// `data` if there is any: rows of texels without padding.
fn texture(
    context: &GpuContext,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    (width, height): (u32, u32),
    data: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test picture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: usage | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if !data.is_empty() {
        let texel_bytes = format
            .block_copy_size(None)
            .expect("an uncompressed format");
        context.queue().write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * texel_bytes),
                rows_per_image: Some(height),
            },
            size,
        );
    }
    texture
}

/// A picture in textures on a context's device: a texture per plane, and Cb
/// and Cr interleaved in another, as NV12 keeps them.
struct Textures {
    planes: [wgpu::TextureView; 3],
    chroma: wgpu::TextureView,
}

impl Textures {
    fn new(context: &GpuContext, layout: &FrameLayout, source: &[Vec<u8>; 3]) -> Self {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |texture: wgpu::Texture| texture.create_view(&Default::default());
        let planes = std::array::from_fn(|i| {
            let size = layout.plane_size(Component::ALL[i]);
            view(texture(
                context,
                wgpu::TextureFormat::R8Unorm,
                usage,
                size,
                &source[i],
            ))
        });
        let interleaved: Vec<u8> = source[1]
            .iter()
            .zip(&source[2])
            .flat_map(|(&cb, &cr)| [cb, cr])
            .collect();
        let size = layout.plane_size(Component::Cb);
        let chroma = view(texture(
            context,
            wgpu::TextureFormat::Rg8Unorm,
            usage,
            size,
            &interleaved,
        ));
        Self { planes, chroma }
    }

    fn planes(&self) -> TexturePicture<'_> {
        TexturePicture::Planes([&self.planes[0], &self.planes[1], &self.planes[2]])
    }

    fn luma_chroma(&self) -> TexturePicture<'_> {
        TexturePicture::LumaChroma {
            luma: &self.planes[0],
            chroma: &self.chroma,
        }
    }
}

/// The sequence number of a frame, from its start of frame header.
fn sequence_of(frame: &[u8]) -> u8 {
    match Header::parse(frame).expect("a start of frame header") {
        Header::StartOfFrame(header) => header.sequence,
        other => panic!("{other:?}"),
    }
}

/// The PSNR of the CPU decode of `frame` against `source`, of all samples.
fn quality(layout: &FrameLayout, source: &[Vec<u8>; 3], frame: &[u8]) -> f64 {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler.push_packet(frame).expect("the frame assembles");
    assert!(assembler.is_ready(false), "the frame is complete");
    let decoded: Vec<u8> = decode_frame(&assembler.frame())
        .expect("the frame decodes")
        .planes
        .iter()
        .flat_map(|plane| plane.to_unorm8())
        .collect();
    psnr(&source.concat(), &decoded)
}

#[test]
fn the_encoders_vectors_come_out_as_the_cpu_made_them_or_as_good() {
    // The GPU's coefficients may differ from the CPU's in the last place, so
    // a frame is the CPU encoder's or differs in a few blocks, never for the
    // worse by more than noise (decision D29).
    let context = gpu_or_return!();
    let (mut frames, mut identical) = (0, 0);
    for vector in vectors() {
        let mut encoder = encoder_for(context, &vector.layout);
        assert_eq!(vector.source.frames.len(), vector.committed.len());
        for (index, (source, expected)) in vector
            .source
            .frames
            .iter()
            .zip(&vector.committed)
            .enumerate()
        {
            let frame = encoder
                .encode(planes(source), vector.budget)
                .expect("the frame fits");
            assert!(
                frame.len() <= vector.budget,
                "{} frame {index}",
                vector.name
            );
            frames += 1;
            if frame == *expected {
                identical += 1;
                continue;
            }
            let (cpu, gpu) = (
                quality(&vector.layout, source, expected),
                quality(&vector.layout, source, &frame),
            );
            println!(
                "{} frame {index}: CPU {} B, {cpu:.3} dB; GPU {} B, {gpu:.3} dB",
                vector.name,
                expected.len(),
                frame.len()
            );
            assert!(gpu >= cpu - 0.05, "{} frame {index}", vector.name);
        }
    }
    println!("{identical} of {frames} frames identical to the CPU encoder's");
    assert!(frames > 10);
}

#[test]
fn the_gpu_decoder_shows_the_gpu_encoders_pictures() {
    let context = gpu_or_return!();
    for (width, height, chroma) in [
        (200, 120, ChromaSubsampling::Yuv420),
        (131, 67, ChromaSubsampling::Yuv444),
    ] {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        // A smooth picture with an edge.
        let source: [Vec<u8>; 3] = std::array::from_fn(|plane| {
            let (w, h) = layout.plane_size(emberwave_core::bitstream::Component::ALL[plane]);
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    (40 + (x * 120 / w + y * 50 / h) as usize + if x > w / 2 { 40 } else { 0 })
                        as u8
                })
                .collect()
        });
        let mut encoder = encoder_for(context, &layout);
        let frame = encoder
            .encode(planes(&source), 200_000)
            .expect("the frame fits");
        let mut decoder = Decoder::new(context, width, height, chroma).expect("a decoder");
        let decoded = decoder.decode(&frame).expect("the GPU decodes it");
        for (plane, (expected, actual)) in source.iter().zip(&decoded.planes).enumerate() {
            let db = psnr(expected, &actual.data);
            assert!(
                db >= 50.0,
                "{width}x{height} {chroma:?} plane {plane}: {db:.2} dB"
            );
        }
    }
}

#[test]
fn errors_are_the_cpu_encoders() {
    let context = gpu_or_return!();
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut gpu = encoder_for(context, &vector.layout);
    let mut cpu = encode::Encoder::new(
        vector.layout.width(),
        vector.layout.height(),
        vector.layout.chroma(),
        Signalling::default(),
    )
    .unwrap();
    let source = &vector.source.frames[0];
    let Err(EncodeError::BudgetTooSmall { minimum, .. }) = cpu.encode(planes(source), 0) else {
        panic!("a frame of 0 bytes");
    };
    for budget in [0, 7, minimum - 1] {
        let expected = cpu.encode(planes(source), budget).expect_err("too small");
        match gpu.encode(planes(source), budget) {
            Err(GpuError::Encode(actual)) => assert_eq!(actual, expected, "{budget} B"),
            other => panic!("{budget} B: {other:?}"),
        }
    }
    let short = &source[1][1..];
    let expected = cpu
        .encode([&source[0], short, &source[2]], 10_000)
        .expect_err("a short plane");
    assert!(matches!(expected, EncodeError::PlaneSize { .. }));
    match gpu.encode([&source[0], short, &source[2]], 10_000) {
        Err(GpuError::Encode(actual)) => assert_eq!(actual, expected),
        other => panic!("{other:?}"),
    }
    // A failure leaves the numbering alone: the next frame is still frame 1.
    let frame = gpu.encode(planes(source), vector.budget).unwrap();
    let sequence = match Header::parse(&frame).expect("a start of frame header") {
        Header::StartOfFrame(header) => header.sequence,
        other => panic!("{other:?}"),
    };
    assert_eq!(sequence, 1, "the first frame after the errors");
}

#[test]
fn a_device_limited_to_webgpu_guarantees_encodes_the_same() {
    let context = gpu_or_return!();
    let options = {
        let mut options = common::options();
        options.strict_webgpu = true;
        options
    };
    let Some(strict) = create_context(&options) else {
        return;
    };
    for vector in vectors().into_iter().take(3) {
        let mut ours = encoder_for(context, &vector.layout);
        let mut limited = encoder_for(&strict, &vector.layout);
        for source in &vector.source.frames {
            let expected = ours.encode(planes(source), vector.budget).unwrap();
            let frame = limited.encode(planes(source), vector.budget).unwrap();
            assert!(frame == expected, "{}", vector.name);
        }
    }
}

#[test]
fn encoding_frame_after_frame_allocates_nothing() {
    // A context of its own: the counters belong to the device.
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let source = &vector.source.frames[0];
    // The largest budget first: the payload buffer, 64 KiB at first, grows
    // to what this picture can need at most, and stays.
    let budgets = [1_000_000, 300, 70_000, 20_000];
    for &budget in &budgets {
        encoder.encode(planes(source), budget).unwrap();
    }
    assert!(
        context.device().get_internal_counters().hal.buffers.read() > 0,
        "wgpu's counters read zero: are they enabled for the tests?"
    );
    let before = live_resources(&context);
    for round in 0..24 {
        let frame = &vector.source.frames[round % vector.source.frames.len()];
        encoder
            .encode(planes(frame), budgets[round % budgets.len()])
            .unwrap();
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}

#[test]
fn a_lost_device_is_an_error_rather_than_a_stale_result() {
    // A context of its own, since destroying the device loses it for good.
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let source = &vector.source.frames[0];
    encoder
        .encode(planes(source), vector.budget)
        .expect("an encode on a working device");
    assert_eq!(encoder.lost(), None);

    context.device().destroy();
    let result = encoder.encode(planes(source), vector.budget);
    assert!(matches!(result, Err(GpuError::DeviceLost(_))), "{result:?}");
    assert!(encoder.lost().is_some());
}

#[test]
fn timing_an_encode_reports_where_the_time_goes() {
    // Timestamp queries are optional, so ask for them and accept their absence.
    let options = {
        let mut options = common::options();
        options.timestamps = true;
        options
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut timed = encoder_for(&context, &vector.layout);
    let mut plain = encoder_for(&context, &vector.layout);
    for source in &vector.source.frames {
        let (frame, timings) = timed.encode_timed(planes(source), vector.budget).unwrap();
        // Timing changes nothing in the frame.
        assert!(frame == plain.encode(planes(source), vector.budget).unwrap());
        assert!(timings.gpu_wall > std::time::Duration::ZERO);
        assert!(timings.prepare > std::time::Duration::ZERO);
        match timings.stages {
            Some(stages) => {
                assert!(context.supports_timestamps());
                // A short stage can be below the timer's resolution and read
                // as zero, but the whole encode cannot, nor the stages of
                // several dispatches: quantisation runs a workgroup per block
                // and rate control four passes and two prefix sums.
                assert!(stages.total() > std::time::Duration::ZERO);
                assert!(stages.quantisation > std::time::Duration::ZERO);
                assert!(stages.rate_control > std::time::Duration::ZERO);
                // The GPU cannot take longer than the wall clock around it.
                assert!(stages.total() <= timings.gpu_wall + timings.prepare);
            }
            // A device that offers timestamps has to deliver them.
            None => assert!(
                !context.supports_timestamps(),
                "the device supports timestamp queries but returned no stage times"
            ),
        }
    }
}

#[test]
fn nothing_is_made_on_a_lost_device() {
    // Making a decoder or an encoder on a lost device used to panic in wgpu
    // (issue #35). Each on a context of its own: the first constructor that
    // fails on a destroyed device records the loss, and a later one would
    // stop at the record before making anything. Before as well as after
    // the context compiled its pipelines, whose failure is then the first.
    for compiled in [false, true] {
        let lost = || {
            let context = create_context(&common::options())?;
            if compiled {
                Decoder::new(&context, 64, 64, ChromaSubsampling::Yuv420).expect("a decoder");
                Encoder::new(
                    &context,
                    64,
                    64,
                    ChromaSubsampling::Yuv420,
                    Signalling::default(),
                )
                .expect("an encoder");
            }
            context.device().destroy();
            Some(context)
        };
        let Some(context) = lost() else {
            return;
        };
        let decoder = Decoder::new(&context, 64, 64, ChromaSubsampling::Yuv420);
        assert!(
            matches!(decoder, Err(GpuError::DeviceLost(_))),
            "compiled {compiled}: {:?}",
            decoder.map(drop)
        );
        let context = lost().expect("a GPU, as before");
        let encoder = Encoder::new(
            &context,
            64,
            64,
            ChromaSubsampling::Yuv420,
            Signalling::default(),
        );
        assert!(
            matches!(encoder, Err(GpuError::DeviceLost(_))),
            "compiled {compiled}: {:?}",
            encoder.map(drop)
        );
    }
}

#[test]
fn pictures_in_textures_encode_as_the_same_planes_do() {
    // gather.wgsl packs the textures into the planes that the upload from
    // CPU memory fills, so the frames are the same bytes, also on a device
    // limited to what WebGPU guarantees (decision D32).
    let context = gpu_or_return!();
    let mut contexts = vec![(context.clone(), usize::MAX)];
    let strict = {
        let mut options = common::options();
        options.strict_webgpu = true;
        options
    };
    if let Some(strict) = create_context(&strict) {
        contexts.push((strict, 3));
    }
    for (context, count) in &contexts {
        for vector in vectors().into_iter().take(*count) {
            let mut from_planes = encoder_for(context, &vector.layout);
            let mut from_textures = encoder_for(context, &vector.layout);
            let mut from_luma_chroma = encoder_for(context, &vector.layout);
            for (index, source) in vector.source.frames.iter().enumerate() {
                let expected = from_planes.encode(planes(source), vector.budget).unwrap();
                let textures = Textures::new(context, &vector.layout, source);
                let frame = from_textures
                    .encode_textures(textures.planes(), vector.budget)
                    .unwrap();
                assert!(frame == expected, "{} frame {index}: planes", vector.name);
                let frame = from_luma_chroma
                    .encode_textures(textures.luma_chroma(), vector.budget)
                    .unwrap();
                assert!(
                    frame == expected,
                    "{} frame {index}: luma and chroma",
                    vector.name
                );
            }
        }
    }
}

#[test]
fn planes_written_in_place_encode_as_uploaded_ones() {
    // On Metal on Apple silicon the encoder writes a picture's planes into
    // the buffer its shaders read, which the gather from textures writes as
    // well; elsewhere, and with `shared_uploads` off, they go through an
    // upload buffer (decision D54). One encoder takes every kind of picture
    // in turn, and its frames are those of an encoder on the upload path.
    let context = gpu_or_return!();
    let options = {
        let mut options = common::options();
        options.shared_uploads = false;
        options
    };
    let Some(copied) = create_context(&options) else {
        return;
    };
    assert!(!copied.shares_uploads());
    for vector in vectors().into_iter().take(3) {
        let mut mixed = encoder_for(context, &vector.layout);
        let mut reference = encoder_for(&copied, &vector.layout);
        for index in 0..8 {
            let source = &vector.source.frames[index % vector.source.frames.len()];
            let expected = reference.encode(planes(source), vector.budget).unwrap();
            let textures = Textures::new(context, &vector.layout, source);
            let mut strides = [0; 3];
            let padded: [Vec<u8>; 3] = std::array::from_fn(|i| {
                let (width, _) = vector.layout.plane_size(Component::ALL[i]);
                let width = width as usize;
                strides[i] = width + 5;
                source[i]
                    .chunks_exact(width)
                    .flat_map(|row| row.iter().copied().chain([0xA5; 5]))
                    .collect()
            });
            let frame = match index % 4 {
                0 => mixed.encode(planes(source), vector.budget),
                1 => mixed.encode_textures(textures.planes(), vector.budget),
                2 => mixed.encode_strided(planes(&padded), strides, vector.budget),
                _ => mixed.encode_textures(textures.luma_chroma(), vector.budget),
            }
            .unwrap();
            assert!(frame == expected, "{} frame {index}", vector.name);
        }
    }
}

#[test]
fn textures_that_cannot_hold_their_planes_are_errors() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let mut encoder = encoder_for(context, &layout);
    let source: [Vec<u8>; 3] = std::array::from_fn(|i| {
        let (width, height) = layout.plane_size(Component::ALL[i]);
        vec![100 + 20 * i as u8; (width * height) as usize]
    });
    let good = Textures::new(context, &layout, &source);
    let bound = wgpu::TextureUsages::TEXTURE_BINDING;
    let view = |texture: wgpu::Texture| texture.create_view(&Default::default());
    let rgba = view(texture(
        context,
        wgpu::TextureFormat::Rgba8Unorm,
        bound,
        (64, 32),
        &[],
    ));
    let small = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        bound,
        (32, 32),
        &[],
    ));
    let unbound = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureUsages::empty(),
        (64, 32),
        &[],
    ));
    let flat_chroma = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        bound,
        (32, 16),
        &[],
    ));
    let cases = [
        (
            "format",
            TexturePicture::Planes([&rgba, &good.planes[1], &good.planes[2]]),
            "the Y texture is Rgba8Unorm rather than R8Unorm or R16Unorm",
        ),
        (
            "size",
            TexturePicture::Planes([&small, &good.planes[1], &good.planes[2]]),
            "the Y texture is 32x32 rather than 64x32",
        ),
        (
            "usage",
            TexturePicture::Planes([&unbound, &good.planes[1], &good.planes[2]]),
            "the Y texture lacks the TEXTURE_BINDING usage",
        ),
        (
            "chroma",
            TexturePicture::LumaChroma {
                luma: &good.planes[0],
                chroma: &flat_chroma,
            },
            "the chroma texture is R8Unorm rather than Rg8Unorm or Rg16Unorm",
        ),
    ];
    for (what, picture, message) in cases {
        match encoder.encode_textures(picture, 100_000) {
            Err(GpuError::Texture(actual)) => assert_eq!(actual, message, "{what}"),
            other => panic!("{what}: {:?}", other.map(drop)),
        }
    }
    let multisampled = context
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("multisampled"),
            size: wgpu::Extent3d {
                width: 64,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: bound | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let picture = TexturePicture::Planes([&multisampled, &good.planes[1], &good.planes[2]]);
    match encoder.encode_textures(picture, 100_000) {
        Err(GpuError::Texture(actual)) => assert_eq!(
            actual,
            "the Y texture has 4 samples per texel rather than 1"
        ),
        other => panic!("samples: {:?}", other.map(drop)),
    }
    // The views are wgpu's to check: an array view of a texture that holds
    // the plane fails in its validation.
    let layers = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("two layers"),
        size: wgpu::Extent3d {
            width: 64,
            height: 32,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: bound | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let array = layers.create_view(&Default::default());
    let picture = TexturePicture::Planes([&array, &good.planes[1], &good.planes[2]]);
    match encoder.encode_textures(picture, 100_000) {
        Err(GpuError::Device(_)) => {}
        other => panic!("an array view: {:?}", other.map(drop)),
    }
    // A failure leaves the numbering alone: the next frame is still frame 1.
    let frame = encoder.encode_textures(good.planes(), 100_000).unwrap();
    assert_eq!(sequence_of(&frame), 1, "the first frame after the errors");

    // A 2D view of one layer of an array texture is a plane like any other.
    context.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &layers,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        &source[0],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64),
            rows_per_image: Some(32),
        },
        wgpu::Extent3d {
            width: 64,
            height: 32,
            depth_or_array_layers: 1,
        },
    );
    let second = layers.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 1,
        array_layer_count: Some(1),
        ..Default::default()
    });
    let picture = TexturePicture::Planes([&second, &good.planes[1], &good.planes[2]]);
    let from_layer = encoder.encode_textures(picture, 100_000).unwrap();
    // Frame 2 of the planes, sequence numbers in every block header included.
    let mut reference = encoder_for(context, &layout);
    reference.encode(planes(&source), 100_000).unwrap();
    let expected = reference.encode(planes(&source), 100_000).unwrap();
    assert!(from_layer == expected, "the frame of a layer's view");
}

#[test]
fn encoding_from_textures_frame_after_frame_allocates_nothing() {
    // A context of its own: the counters belong to the device. The bind
    // group of each frame's textures goes when the frame is done.
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let textures: Vec<Textures> = vector
        .source
        .frames
        .iter()
        .map(|source| Textures::new(&context, &vector.layout, source))
        .collect();
    let mut encode = |round: usize| {
        let frame = &textures[round % textures.len()];
        // Each frame in both layouts, in turn.
        let picture = if (round / textures.len()).is_multiple_of(2) {
            frame.planes()
        } else {
            frame.luma_chroma()
        };
        encoder.encode_textures(picture, vector.budget).unwrap();
    };
    for round in 0..4 {
        encode(round);
    }
    let before = live_resources(&context);
    for round in 0..24 {
        encode(round);
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}

/// A vector's picture as RGB: four bytes a pixel in `order`, rows `stride`
/// bytes apart with 0x5A between them, and the fourth byte varied.
fn rgb_of(layout: &FrameLayout, source: &[Vec<u8>; 3], order: RgbOrder, stride: usize) -> Vec<u8> {
    let (width, height) = (layout.width() as usize, layout.height() as usize);
    let rgb = yuv_to_rgb8(
        planes(source),
        layout.width(),
        layout.height(),
        layout.chroma(),
        &Signalling::default().color_params(),
    )
    .expect("the planes fit");
    let mut bytes = vec![0x5A; stride * (height - 1) + 4 * width];
    for (i, pixel) in rgb.chunks_exact(3).enumerate() {
        let (x, y) = (i % width, i / width);
        let [r, g, b] = [pixel[0], pixel[1], pixel[2]];
        let fourth = (x * 7 + y * 13) as u8;
        let ordered = match order {
            RgbOrder::Rgba => [r, g, b, fourth],
            RgbOrder::Bgra => [b, g, r, fourth],
            other => panic!("{other:?}"),
        };
        bytes[y * stride + 4 * x..][..4].copy_from_slice(&ordered);
    }
    bytes
}

#[test]
fn pictures_in_rgb_encode_as_the_planes_they_convert_to_do() {
    // gather_rgb.wgsl converts as rgb_to_ycbcr8 does, to the last bit, so a
    // picture in an RGB texture of either order, or as RGB in CPU memory,
    // encodes to the frame of the planes rgb_to_ycbcr8 makes of it, also on
    // a device limited to what WebGPU guarantees (decision D65).
    let context = gpu_or_return!();
    let mut contexts = vec![(context.clone(), usize::MAX)];
    let strict = {
        let mut options = common::options();
        options.strict_webgpu = true;
        options
    };
    if let Some(strict) = create_context(&strict) {
        contexts.push((strict, 2));
    }
    let bound = wgpu::TextureUsages::TEXTURE_BINDING;
    for (context, count) in &contexts {
        for vector in vectors().into_iter().take(*count) {
            let layout = &vector.layout;
            let (width, height) = (layout.width(), layout.height());
            // An encoder for each way, so that their frames are numbered
            // alike, and one that takes the two orders in turn.
            let mut from_planes = encoder_for(context, layout);
            let mut from_rgb: Vec<Encoder> = (0..4).map(|_| encoder_for(context, layout)).collect();
            let mut alternating = encoder_for(context, layout);
            let cpu = || {
                encode::Encoder::new(width, height, layout.chroma(), Signalling::default()).unwrap()
            };
            let (mut cpu_planes, mut cpu_rgb) = (cpu(), cpu());
            for (index, source) in vector.source.frames.iter().enumerate() {
                let row = 4 * width as usize;
                let rgba = rgb_of(layout, source, RgbOrder::Rgba, row);
                let bgra = rgb_of(layout, source, RgbOrder::Bgra, row);
                let converted = rgb_to_ycbcr8(
                    &rgba,
                    row,
                    RgbOrder::Rgba,
                    width,
                    height,
                    layout.chroma(),
                    &Signalling::default().color_params(),
                )
                .unwrap();
                let expected = from_planes
                    .encode(planes(&converted), vector.budget)
                    .unwrap();
                let what = |how: &str| format!("{} frame {index}: {how}", vector.name);
                let [rgba_texture, bgra_texture, bgra_memory, rgba_memory] = &mut from_rgb[..]
                else {
                    unreachable!()
                };
                for (how, encoder, format, bytes) in [
                    (
                        "RGBA texture",
                        rgba_texture,
                        wgpu::TextureFormat::Rgba8Unorm,
                        &rgba,
                    ),
                    (
                        "BGRA texture",
                        bgra_texture,
                        wgpu::TextureFormat::Bgra8Unorm,
                        &bgra,
                    ),
                ] {
                    let view = texture(context, format, bound, (width, height), bytes)
                        .create_view(&Default::default());
                    let frame = encoder
                        .encode_textures(TexturePicture::Rgb(&view), vector.budget)
                        .unwrap();
                    assert!(frame == expected, "{}", what(how));
                }
                let stride = row + 20;
                for (encoder, order) in
                    [(bgra_memory, RgbOrder::Bgra), (rgba_memory, RgbOrder::Rgba)]
                {
                    let padded = rgb_of(layout, source, order, stride);
                    let frame = encoder
                        .encode_rgb(&padded, stride, order, vector.budget)
                        .unwrap();
                    assert!(
                        frame == expected,
                        "{}",
                        what(&format!("{order:?} in CPU memory"))
                    );
                }
                let order = if index % 2 == 0 {
                    RgbOrder::Rgba
                } else {
                    RgbOrder::Bgra
                };
                let picture = rgb_of(layout, source, order, row);
                let frame = alternating
                    .encode_rgb(&picture, row, order, vector.budget)
                    .unwrap();
                assert!(frame == expected, "{}", what(&format!("{order:?} in turn")));
                // The CPU encoder converts alike.
                let expected = cpu_planes
                    .encode(planes(&converted), vector.budget)
                    .unwrap();
                let frame = cpu_rgb
                    .encode_rgb(&bgra, row, RgbOrder::Bgra, vector.budget)
                    .unwrap();
                assert!(frame == expected, "{}", what("the CPU encoder"));
            }
        }
    }
}

#[test]
fn rgb_pictures_that_do_not_fit_are_errors() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let mut encoder = encoder_for(context, &layout);
    let bound = wgpu::TextureUsages::TEXTURE_BINDING;
    let view =
        |format, size| texture(context, format, bound, size, &[]).create_view(&Default::default());
    let cases = [
        (
            view(wgpu::TextureFormat::R8Unorm, (64, 32)),
            "the RGB texture is R8Unorm rather than Rgba8Unorm or Bgra8Unorm",
        ),
        // Its reads would undo the transfer function.
        (
            view(wgpu::TextureFormat::Bgra8UnormSrgb, (64, 32)),
            "the RGB texture is Bgra8UnormSrgb rather than Rgba8Unorm or Bgra8Unorm",
        ),
        (
            view(wgpu::TextureFormat::Rgba8Unorm, (32, 32)),
            "the RGB texture is 32x32 rather than 64x32",
        ),
    ];
    for (view, message) in &cases {
        match encoder.encode_textures(TexturePicture::Rgb(view), 100_000) {
            Err(GpuError::Texture(actual)) => assert_eq!(&actual, message),
            other => panic!("{message}: {:?}", other.map(drop)),
        }
    }
    let bytes = vec![128u8; 64 * 4 * 32];
    let short = &bytes[..bytes.len() - 1];
    match encoder.encode_rgb(short, 256, RgbOrder::Bgra, 100_000) {
        Err(GpuError::Encode(EncodeError::Picture(ColorError::PictureTooSmall {
            needed: 8192,
            actual: 8191,
        }))) => {}
        other => panic!("a short picture: {:?}", other.map(drop)),
    }
    match encoder.encode_rgb(&bytes, 255, RgbOrder::Bgra, 100_000) {
        Err(GpuError::Encode(EncodeError::Picture(ColorError::StrideTooSmall {
            stride: 255,
            row: 256,
        }))) => {}
        other => panic!("a short stride: {:?}", other.map(drop)),
    }
    // A failure leaves the numbering alone: the next frame is still frame 1.
    let frame = encoder
        .encode_rgb(&bytes, 256, RgbOrder::Bgra, 100_000)
        .unwrap();
    assert_eq!(sequence_of(&frame), 1, "the first frame after the errors");
}

#[test]
fn encoding_rgb_frame_after_frame_allocates_nothing() {
    // A context of its own, as for textures: the texture that RGB in CPU
    // memory goes into is kept, and made again only for the other order.
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let layout = &vector.layout;
    let row = 4 * layout.width() as usize;
    let pictures: Vec<Vec<u8>> = vector
        .source
        .frames
        .iter()
        .map(|source| rgb_of(layout, source, RgbOrder::Bgra, row))
        .collect();
    let view = texture(
        &context,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::TEXTURE_BINDING,
        (layout.width(), layout.height()),
        &pictures[0],
    )
    .create_view(&Default::default());
    let mut encoder = encoder_for(&context, layout);
    let mut encode = |round: usize| {
        if round % 3 == 2 {
            encoder.encode_textures(TexturePicture::Rgb(&view), vector.budget)
        } else {
            let picture = &pictures[round % pictures.len()];
            encoder.encode_rgb(picture, row, RgbOrder::Bgra, vector.budget)
        }
        .unwrap();
    };
    for round in 0..4 {
        encode(round);
    }
    let before = live_resources(&context);
    for round in 0..24 {
        encode(round);
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
    // The other order replaces the buffer and the texture rather than
    // adding others. Their memory tells: wgpu-hal 30's Vulkan backend counts
    // the textures it destroys but not those it makes.
    let memory = |context: &GpuContext| {
        let hal = context.device().get_internal_counters().hal;
        (
            hal.buffers.read(),
            hal.buffer_memory.read(),
            hal.texture_memory.read(),
        )
    };
    let before = memory(&context);
    encoder
        .encode_rgb(&pictures[0], row, RgbOrder::Rgba, vector.budget)
        .unwrap();
    assert_eq!(memory(&context), before, "after the other order");
}

#[test]
fn one_row_rgb_pictures_take_any_stride() {
    // A picture of one row has no stride to keep: zero, less than the row,
    // or past what a copy takes, it encodes as the row's planes do.
    let context = gpu_or_return!();
    let layout = FrameLayout::new(64, 1, ChromaSubsampling::Yuv444).unwrap();
    let row: Vec<u8> = (0..64 * 4).map(|i| (i * 37 % 251) as u8).collect();
    let converted = rgb_to_ycbcr8(
        &row,
        256,
        RgbOrder::Bgra,
        64,
        1,
        ChromaSubsampling::Yuv444,
        &Signalling::default().color_params(),
    )
    .unwrap();
    for stride in [0, 255, 256, 1 << 33] {
        let expected = encoder_for(context, &layout)
            .encode(planes(&converted), 4_000)
            .unwrap();
        let frame = encoder_for(context, &layout)
            .encode_rgb(&row, stride, RgbOrder::Bgra, 4_000)
            .unwrap();
        assert!(frame == expected, "stride {stride}");
    }
}

#[test]
fn planes_with_padded_rows_encode_as_back_to_back_ones_do() {
    // Rows `extra` bytes longer than the plane, filled with what a capture
    // might leave there, which the encoder must not read.
    let context = gpu_or_return!();
    for vector in vectors().into_iter().take(3) {
        let mut tight = encoder_for(context, &vector.layout);
        let mut strided = encoder_for(context, &vector.layout);
        for (index, source) in vector.source.frames.iter().enumerate() {
            let extra = 13 + index;
            let mut strides = [0; 3];
            let padded: [Vec<u8>; 3] = std::array::from_fn(|i| {
                let (width, _) = vector.layout.plane_size(Component::ALL[i]);
                let width = width as usize;
                strides[i] = width + extra;
                source[i]
                    .chunks_exact(width)
                    .flat_map(|row| {
                        row.iter()
                            .copied()
                            .chain((0..extra).map(|j| (j * 37) as u8))
                    })
                    .collect()
            });
            let expected = tight.encode(planes(source), vector.budget).unwrap();
            let frame = strided
                .encode_strided(planes(&padded), strides, vector.budget)
                .unwrap();
            assert!(frame == expected, "{} frame {index}", vector.name);
        }
    }

    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let mut encoder = encoder_for(context, &layout);
    let source: [Vec<u8>; 3] = std::array::from_fn(|i| {
        let (width, height) = layout.plane_size(Component::ALL[i]);
        vec![90; (width * height) as usize]
    });
    match encoder.encode_strided(planes(&source), [64, 31, 32], 10_000) {
        Err(GpuError::Encode(EncodeError::PlaneStride {
            component: Component::Cb,
            stride: 31,
            width: 32,
        })) => {}
        other => panic!("a stride below the width: {:?}", other.map(drop)),
    }
    // The last row may end at the plane's width; a byte short is an error.
    let short = &source[2][..32 * 15 + 31];
    match encoder.encode_strided([&source[0], &source[1], short], [64, 32, 32], 10_000) {
        Err(GpuError::Encode(EncodeError::PlaneSize {
            component: Component::Cr,
            expected: 512,
            actual: 511,
        })) => {}
        other => panic!("a short plane: {:?}", other.map(drop)),
    }
    let frame = encoder
        .encode_strided(planes(&source), [64, 32, 32], 10_000)
        .unwrap();
    assert_eq!(sequence_of(&frame), 1, "the first frame after the errors");
    // With rows wider than the plane, the last row also ends at the
    // plane's width: (height - 1) * stride + width bytes suffice.
    let stride = 40;
    let exact: Vec<u8> = source[1]
        .chunks_exact(32)
        .enumerate()
        .flat_map(|(row, samples)| {
            let padding = if row == 15 { 0 } else { stride - 32 };
            samples
                .iter()
                .copied()
                .chain(std::iter::repeat_n(7, padding))
        })
        .collect();
    assert_eq!(exact.len(), 15 * stride + 32);
    let padded = encoder
        .encode_strided([&source[0], &exact, &source[2]], [64, stride, 32], 10_000)
        .unwrap();
    // Frame 2 of the same picture, sequence numbers in every block header
    // included.
    let mut reference = encoder_for(context, &layout);
    reference.encode(planes(&source), 10_000).unwrap();
    let tight = reference.encode(planes(&source), 10_000).unwrap();
    assert!(
        padded == tight,
        "the frame of the exactly sized padded plane"
    );
}

#[test]
fn a_frame_that_outgrows_the_payload_buffer_decodes_as_in_a_new_decoder() {
    // A decoder's payload buffer, and the upload buffer it keeps for it
    // (decision D34), start at 64 KiB and grow with the frames. A large frame
    // after a small one decodes as in a decoder that saw nothing before, and
    // so does the small one after the large.
    let context = gpu_or_return!();
    let (width, height) = (640, 360);
    let chroma = ChromaSubsampling::Yuv444;
    let mut state = 0x2545_f491_u32;
    let noise: [Vec<u8>; 3] = std::array::from_fn(|_| {
        (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    });
    let mut cpu = encode::Encoder::new(width, height, chroma, Signalling::default()).unwrap();
    let small = cpu.encode(planes(&noise), 20_000).unwrap();
    let large = cpu.encode(planes(&noise), 600_000).unwrap();
    assert!(large.len() > 4 * 64 * 1024, "{} bytes", large.len());

    let mut used = Decoder::new(context, width, height, chroma).unwrap();
    let fresh = || Decoder::new(context, width, height, chroma).unwrap();
    used.decode(&small).unwrap();
    for frame in [&large, &small] {
        let after_others = used.decode(frame).unwrap();
        let first = fresh().decode(frame).unwrap();
        for (plane, (a, b)) in after_others.planes.iter().zip(&first.planes).enumerate() {
            assert!(a.data == b.data, "{} bytes, plane {plane}", frame.len());
        }
    }
}

/// 16-bit UNORM picture textures (3-plane `R16Unorm` and P010 `R16Unorm` + `Rg16Unorm`).
struct Textures16 {
    _owned: Vec<wgpu::Texture>,
    planes: [wgpu::TextureView; 3],
    chroma: wgpu::TextureView,
}

impl Textures16 {
    fn new(context: &GpuContext, layout: &FrameLayout, source: &[Vec<u16>; 3]) -> Self {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        let sizes = Component::ALL.map(|c| layout.plane_size(c));
        let bytes: [Vec<u8>; 3] =
            std::array::from_fn(|i| source[i].iter().flat_map(|s| s.to_le_bytes()).collect());
        let mut owned: Vec<wgpu::Texture> = (0..3)
            .map(|i| {
                texture(
                    context,
                    emberwave_core::gpu::PLANE_FORMAT_R16,
                    usage,
                    sizes[i],
                    &bytes[i],
                )
            })
            .collect();
        let interleaved: Vec<u8> = source[1]
            .iter()
            .zip(&source[2])
            .flat_map(|(&cb, &cr)| {
                let [cb0, cb1] = cb.to_le_bytes();
                let [cr0, cr1] = cr.to_le_bytes();
                [cb0, cb1, cr0, cr1]
            })
            .collect();
        owned.push(texture(
            context,
            emberwave_core::gpu::CHROMA_FORMAT_RG16,
            usage,
            sizes[1],
            &interleaved,
        ));
        let planes = std::array::from_fn(|i| owned[i].create_view(&Default::default()));
        let chroma = owned[3].create_view(&Default::default());
        Self {
            _owned: owned,
            planes,
            chroma,
        }
    }

    fn planes(&self) -> TexturePicture<'_> {
        TexturePicture::Planes([&self.planes[0], &self.planes[1], &self.planes[2]])
    }

    fn p010(&self) -> TexturePicture<'_> {
        TexturePicture::LumaChroma {
            luma: &self.planes[0],
            chroma: &self.chroma,
        }
    }
}

fn gradient_u16(layout: &FrameLayout, seed: u32, p010: bool) -> [Vec<u16>; 3] {
    Component::ALL.map(|component| {
        let (width, height) = layout.plane_size(component);
        let base = 64 + component.index() as u32 * 100 + seed * 17;
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                let code = (base + x * 4 + y * 3) % 940;
                if p010 {
                    (code as u16) << 6
                } else {
                    emberwave_core::cpu::unorm16(code as f32 / 1023.0)
                }
            })
            .collect()
    })
}

fn read_u16_texture(context: &GpuContext, texture: &wgpu::Texture) -> Vec<u16> {
    let size = texture.size();
    let row_bytes = size.width * 2;
    let aligned = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("r16 readback"),
        size: u64::from(aligned) * u64::from(size.height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    context.queue().submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((size.width * size.height) as usize);
    for row in 0..size.height as usize {
        let bytes = &mapped[row * aligned as usize..][..row_bytes as usize];
        for pair in bytes.chunks_exact(2) {
            out.push(u16::from_le_bytes([pair[0], pair[1]]));
        }
    }
    drop(mapped);
    staging.unmap();
    out
}

#[test]
fn u16_planes_strided_and_textures_including_p010_encode_identically() {
    let context = gpu_or_return!();
    let has_r16 = context.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16)
        && context.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16);
    for (width, height, chroma) in [
        (200, 120, ChromaSubsampling::Yuv420),
        (131, 67, ChromaSubsampling::Yuv444),
    ] {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let mut from_u16 = encoder_for(context, &layout);
        let mut from_strided = encoder_for(context, &layout);
        let mut from_le = encoder_for(context, &layout);
        let mut from_timed = encoder_for(context, &layout);
        let mut from_planes_tex = encoder_for(context, &layout);
        let mut from_p010_tex = encoder_for(context, &layout);

        for (seed, p010) in [(0, false), (1, true)] {
            let src = gradient_u16(&layout, seed, p010);
            let expected = from_u16
                .encode_u16([&src[0], &src[1], &src[2]], 30_000)
                .unwrap();

            // Sample-strided u16 planes with trailing garbage samples.
            let mut strides_u16 = [0usize; 3];
            let padded_u16: [Vec<u16>; 3] = std::array::from_fn(|i| {
                let w = layout.plane_size(Component::ALL[i]).0 as usize;
                let extra = 5 + i;
                strides_u16[i] = w + extra;
                src[i]
                    .chunks_exact(w)
                    .flat_map(|row| {
                        row.iter()
                            .copied()
                            .chain(std::iter::repeat_n(0xBEEF, extra))
                    })
                    .collect()
            });
            let got_strided = from_strided
                .encode_u16_strided(
                    [&padded_u16[0], &padded_u16[1], &padded_u16[2]],
                    strides_u16,
                    30_000,
                )
                .unwrap();
            assert_eq!(got_strided, expected);

            // Byte-strided LE planes (including odd byte stride!).
            let mut strides_le = [0usize; 3];
            let padded_le: [Vec<u8>; 3] = std::array::from_fn(|i| {
                let w = layout.plane_size(Component::ALL[i]).0 as usize;
                let extra = 3 + i * 2;
                strides_le[i] = w * 2 + extra;
                src[i]
                    .chunks_exact(w)
                    .flat_map(|row| {
                        row.iter()
                            .flat_map(|s| s.to_le_bytes())
                            .chain(std::iter::repeat_n(0xA5, extra))
                    })
                    .collect()
            });
            let got_le = from_le
                .encode_u16_le_strided(
                    [&padded_le[0], &padded_le[1], &padded_le[2]],
                    strides_le,
                    30_000,
                )
                .unwrap();
            assert_eq!(got_le, expected);

            let (got_timed, timings) = from_timed
                .encode_u16_timed([&src[0], &src[1], &src[2]], 30_000)
                .unwrap();
            assert_eq!(got_timed, expected);
            assert!(
                timings.gpu_wall > std::time::Duration::ZERO
                    || timings.prepare >= std::time::Duration::ZERO
            );

            if has_r16 {
                let textures = Textures16::new(context, &layout, &src);
                let got_planes = from_planes_tex
                    .encode_textures(textures.planes(), 30_000)
                    .unwrap();
                assert_eq!(
                    got_planes, expected,
                    "{width}x{height} r16 planes seed {seed}"
                );
                let got_p010 = from_p010_tex
                    .encode_textures(textures.p010(), 30_000)
                    .unwrap();
                assert_eq!(got_p010, expected, "{width}x{height} p010 seed {seed}");
            }
        }
    }
}

#[test]
fn u16_encode_to_r16_decode_roundtrip_preserves_10bit_precision() {
    let context = gpu_or_return!();
    if !context.supports_plane_format(emberwave_core::gpu::PLANE_FORMAT_R16)
        || !context.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16)
        || !context.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16)
    {
        return;
    }
    let layout = FrameLayout::new(192, 136, ChromaSubsampling::Yuv420).unwrap();
    let src16 = gradient_u16(&layout, 0, false);
    let src8: [Vec<u8>; 3] = std::array::from_fn(|i| {
        src16[i]
            .iter()
            .map(|&s| ((u32::from(s) * 255 + 32767) / 65535) as u8)
            .collect()
    });

    let mut enc16 = encoder_for(context, &layout);
    let textures = Textures16::new(context, &layout, &src16);
    let frame16 = enc16.encode_textures(textures.p010(), 80_000).unwrap();

    let mut enc8 = encoder_for(context, &layout);
    let frame8 = enc8.encode(planes(&src8), 80_000).unwrap();

    let mut decoder = Decoder::new(context, 192, 136, ChromaSubsampling::Yuv420).unwrap();
    let out_tex = Component::ALL.map(|c| {
        decoder
            .create_plane_texture_with_format(c, emberwave_core::gpu::PLANE_FORMAT_R16)
            .unwrap()
    });
    let out_views: [_; 3] = std::array::from_fn(|i| out_tex[i].create_view(&Default::default()));

    let decode_to_u16 = |decoder: &mut Decoder, frame: &[u8]| -> [Vec<u16>; 3] {
        decoder.clear();
        decoder.push_packet(frame).unwrap();
        decoder
            .decode_assembled_to_plane_textures([&out_views[0], &out_views[1], &out_views[2]])
            .unwrap();
        std::array::from_fn(|i| read_u16_texture(context, &out_tex[i]))
    };

    let dec16 = decode_to_u16(&mut decoder, &frame16);
    let dec8 = decode_to_u16(&mut decoder, &frame8);

    let psnr16_fn = |a: &[u16], b: &[u16]| -> f64 {
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        10.0 * (65535.0 * 65535.0 / mse.max(1e-12)).log10()
    };
    let psnr16 = (0..3)
        .map(|i| psnr16_fn(&src16[i], &dec16[i]))
        .fold(f64::INFINITY, f64::min);
    let psnr8 = (0..3)
        .map(|i| psnr16_fn(&src16[i], &dec8[i]))
        .fold(f64::INFINITY, f64::min);
    assert!(psnr16 > 60.0, "16-bit GPU round-trip PSNR {psnr16:.2} dB");
    assert!(
        psnr16 > psnr8 + 5.0,
        "16-bit GPU round-trip ({psnr16:.2} dB) must beat 8-bit ({psnr8:.2} dB)"
    );
}

#[test]
fn u16_unsupported_feature_and_invalid_textures_are_rejected() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();

    // Even under strict_webgpu or r16_planes = false (no TEXTURE_FORMAT_16BIT_NORM),
    // CPU encode_u16 works, while supports_source_format reports false for R16/Rg16.
    for mut opts in [
        {
            let mut o = common::options();
            o.strict_webgpu = true;
            o
        },
        {
            let mut o = common::options();
            o.r16_planes = false;
            o
        },
    ] {
        opts.timestamps = false;
        if let Some(limited) = create_context(&opts) {
            assert!(!limited.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16));
            assert!(!limited.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16));
            let mut enc = encoder_for(&limited, &layout);
            assert!(!enc.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16));
            assert!(!enc.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16));
            let src = gradient_u16(&layout, 0, false);
            let frame = enc
                .encode_u16([&src[0], &src[1], &src[2]], 10_000)
                .expect("CPU encode_u16 works without TEXTURE_FORMAT_16BIT_NORM");
            assert_eq!(sequence_of(&frame), 1);
        }
    }

    // Bad strides/sizes and mixed 8-bit/16-bit textures are rejected without
    // advancing the sequence number.
    let mut encoder = encoder_for(context, &layout);
    let src16 = gradient_u16(&layout, 0, false);
    assert!(matches!(
        encoder.encode_u16_strided([&src16[0], &src16[1], &src16[2]], [64, 31, 32], 10_000),
        Err(GpuError::Encode(EncodeError::PlaneStride {
            component: Component::Cb,
            stride: 31,
            width: 32,
        }))
    ));
    let bytes_le: [Vec<u8>; 3] =
        std::array::from_fn(|i| src16[i].iter().flat_map(|s| s.to_le_bytes()).collect());
    assert!(matches!(
        encoder.encode_u16_le_strided(
            [&bytes_le[0], &bytes_le[1], &bytes_le[2]],
            [127, 64, 64],
            10_000
        ),
        Err(GpuError::Encode(EncodeError::PlaneStride {
            component: Component::Y,
            stride: 127,
            width: 128,
        }))
    ));

    if context.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16)
        && context.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16)
    {
        let src8: [Vec<u8>; 3] = std::array::from_fn(|i| {
            let (w, h) = layout.plane_size(Component::ALL[i]);
            vec![128u8; (w * h) as usize]
        });
        let tex8 = Textures::new(context, &layout, &src8);
        let tex16 = Textures16::new(context, &layout, &src16);
        for mixed in [
            TexturePicture::Planes([&tex16.planes[0], &tex8.planes[1], &tex8.planes[2]]),
            TexturePicture::Planes([&tex8.planes[0], &tex16.planes[1], &tex8.planes[2]]),
            TexturePicture::LumaChroma {
                luma: &tex16.planes[0],
                chroma: &tex8.chroma,
            },
            TexturePicture::LumaChroma {
                luma: &tex8.planes[0],
                chroma: &tex16.chroma,
            },
        ] {
            assert!(
                matches!(
                    encoder.encode_textures(mixed, 10_000),
                    Err(GpuError::Texture(_))
                ),
                "mixed 8-bit/16-bit textures must fail"
            );
        }
    }

    let first = encoder
        .encode_u16([&src16[0], &src16[1], &src16[2]], 10_000)
        .unwrap();
    assert_eq!(sequence_of(&first), 1);
}

#[test]
fn encoding_u16_frame_after_frame_allocates_nothing() {
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let layout = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
    let src = gradient_u16(&layout, 0, true);
    let has_r16 = context.supports_source_format(emberwave_core::gpu::PLANE_FORMAT_R16)
        && context.supports_source_format(emberwave_core::gpu::CHROMA_FORMAT_RG16);
    let tex16 = has_r16.then(|| Textures16::new(&context, &layout, &src));
    let mut encoder = encoder_for(&context, &layout);

    let mut run_round = |round: usize| {
        if let Some(tex) = &tex16 {
            match round % 3 {
                0 => encoder.encode_u16([&src[0], &src[1], &src[2]], 20_000),
                1 => encoder.encode_textures(tex.planes(), 20_000),
                _ => encoder.encode_textures(tex.p010(), 20_000),
            }
        } else {
            encoder.encode_u16([&src[0], &src[1], &src[2]], 20_000)
        }
        .unwrap();
    };
    for round in 0..6 {
        run_round(round);
    }
    let before = live_resources(&context);
    for round in 0..24 {
        run_round(round);
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}
