//! A minimal reader and writer of 8-bit and 10-bit planar YUV4MPEG2 (`.y4m`) files.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use emberwave_core::bitstream::{ChromaSiting, ChromaSubsampling, YcbcrRange};
use emberwave_core::cpu::unorm16;

use crate::Result;

/// Sample bit depth supported by the Y4M reader and writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 8-bit unsigned samples (`C420`, `C420jpeg`, `C420mpeg2`, `C420paldv`, `C444`).
    Eight,
    /// 10-bit samples in little-endian 16-bit words (`C420p10`, `C444p10`).
    Ten,
}

impl BitDepth {
    /// Number of bits per sample (`8` or `10`).
    pub fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
        }
    }

    /// Number of bytes per sample in a Y4M plane (`1` or `2`).
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Ten => 2,
        }
    }
}

/// Converts a 10-bit code value (`0..=1023`) to 16-bit UNORM (`0..=65535`).
pub fn u10_to_unorm16(sample: u16) -> Result<u16> {
    if sample > 1023 {
        return Err(format!("10-bit Y4M sample {sample} exceeds 1023").into());
    }
    Ok(unorm16(f32::from(sample) / 1023.0))
}

/// Converts a 16-bit UNORM sample (`0..=65535`) to the nearest 10-bit code
/// value (`0..=1023`). Since `65535` is odd, `sample * 1023 / 65535` never hits
/// an exact half-integer tie, so `(sample * 1023 + 32767) / 65535` matches exact
/// round-to-nearest for every `u16`.
pub fn unorm16_to_u10(sample: u16) -> u16 {
    ((u32::from(sample) * 1023 + 32_767) / 65_535) as u16
}

/// Converts a 16-bit UNORM sample (`0..=65535`) from a stream with
/// `source_bit_depth` and `range` to a 10-bit code value (`0..=1023`).
///
/// Limited-range 8-bit (`c8 / 255`) and 16-bit (`c16 / 65535`) streams place
/// nominal levels at multiples of `4 / 1020` and `64 / 65535` rather than
/// `1 / 1023`, so convert those scales explicitly to preserve `64..=940`
/// (chroma `64..=960`, neutral `512`).
pub fn unorm16_to_u10_with_range(sample: u16, source_bit_depth: u8, range: YcbcrRange) -> u16 {
    if range == YcbcrRange::Limited {
        match source_bit_depth {
            8 => ((u32::from(sample) * 1020 + 32_767) / 65_535).min(1023) as u16,
            16 => {
                let s = u32::from(sample);
                ((s + 31 + ((s >> 6) & 1)) >> 6).min(1023) as u16
            }
            _ => unorm16_to_u10(sample),
        }
    } else {
        unorm16_to_u10(sample)
    }
}

/// Converts a 16-bit UNORM sample (`0..=65535`) from a 10-bit limited-range
/// stream to an 8-bit limited-range sample (`c10 / 4`, rounded ties to even,
/// clamped to `0..=255`), preserving nominal levels `64 -> 16`, `512 -> 128`,
/// `940 -> 235`, and `960 -> 240`.
pub fn unorm16_to_u8_limited_from_u10(sample: u16) -> u8 {
    let c10 = u32::from(unorm16_to_u10(sample));
    ((c10 + 1 + ((c10 >> 2) & 1)) >> 2).min(255) as u8
}

/// Fallback conversion from an 8-bit UNORM sample of a 10-bit limited-range
/// stream (`c10 / 1023 * 255`) to 8-bit limited range when `R16Unorm` is
/// unavailable on the adapter.
pub fn unorm8_to_u8_limited_from_u10(sample: u8) -> u8 {
    ((u32::from(sample) * 1023 + 510) / 1020).min(255) as u8
}

/// Decodes a little-endian 10-bit Y4M plane buffer into `u16` code values in `0..=1023`.
pub fn decode_u10_le_plane(bytes: &[u8]) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("10-bit Y4M plane has odd byte length {}", bytes.len()).into());
    }
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let sample = u16::from_le_bytes([pair[0], pair[1]]);
        if sample > 1023 {
            return Err(format!("10-bit Y4M sample {sample} exceeds 1023").into());
        }
        samples.push(sample);
    }
    Ok(samples)
}

/// Converts a little-endian 10-bit Y4M plane buffer into 16-bit UNORM samples (`0..=65535`).
pub fn u10_plane_to_unorm16(bytes: &[u8]) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("10-bit Y4M plane has odd byte length {}", bytes.len()).into());
    }
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let sample = u16::from_le_bytes([pair[0], pair[1]]);
        samples.push(u10_to_unorm16(sample)?);
    }
    Ok(samples)
}

/// The frame rate a Y4M file gets when the stream has none: Y4M has no word
/// for "unrelated images".
pub const DEFAULT_FRAME_RATE: (u32, u32) = (30, 1);

/// The longest header or frame line read: far more than any has, and a
/// bound on what a file that is not Y4M can make the reader hold.
const MAX_LINE: u64 = 64 * 1024;

/// What the header of a Y4M file says about its pictures.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub chroma: ChromaSubsampling,
    /// Where 4:2:0 chroma sits; Y4M has a tag for each case.
    pub siting: ChromaSiting,
    /// Numerator and denominator. A zero in either means the stream has none.
    pub frame_rate: (u32, u32),
    pub range: YcbcrRange,
    pub bit_depth: BitDepth,
}

impl StreamInfo {
    /// Whether the stream carries a frame rate of its own.
    pub fn has_frame_rate(&self) -> bool {
        self.frame_rate.0 != 0 && self.frame_rate.1 != 0
    }

    /// The header line, without its newline.
    fn header_line(&self) -> String {
        let (num, den) = if self.has_frame_rate() {
            self.frame_rate
        } else {
            DEFAULT_FRAME_RATE
        };
        let range = match self.range {
            YcbcrRange::Full => "FULL",
            YcbcrRange::Limited => "LIMITED",
        };
        // `420jpeg` has chroma centred between the luma samples both ways,
        // `420mpeg2` has it on the left luma column and centred vertically.
        // Y4M defines no siting-specific `C420p10` tag, so 10-bit 4:2:0 always
        // writes `C420p10` (callers warn if a left-sited stream is written).
        let chroma = match (self.bit_depth, self.chroma, self.siting) {
            (BitDepth::Eight, ChromaSubsampling::Yuv444, _) => "444",
            (BitDepth::Eight, ChromaSubsampling::Yuv420, ChromaSiting::Center) => "420jpeg",
            (BitDepth::Eight, ChromaSubsampling::Yuv420, ChromaSiting::Left) => "420mpeg2",
            (BitDepth::Ten, ChromaSubsampling::Yuv444, _) => "444p10",
            (BitDepth::Ten, ChromaSubsampling::Yuv420, _) => "420p10",
        };
        format!(
            "YUV4MPEG2 W{} H{} F{num}:{den} Ip A1:1 XCOLORRANGE={range} C{chroma}",
            self.width, self.height
        )
    }
}

/// Writes frames of 8-bit or 10-bit planar YUV.
pub struct Writer {
    out: BufWriter<File>,
}

impl Writer {
    /// Creates the file and writes the stream header.
    pub fn create(path: &Path, info: &StreamInfo) -> Result<Self> {
        let file = File::create(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut out = BufWriter::new(file);
        writeln!(out, "{}", info.header_line())?;
        Ok(Self { out })
    }

    /// Appends one frame.
    pub fn write_frame(&mut self, planes: [&[u8]; 3]) -> Result<()> {
        self.out.write_all(b"FRAME\n")?;
        for plane in planes {
            self.out.write_all(plane)?;
        }
        Ok(())
    }

    /// Flushes everything to disk.
    pub fn finish(mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

/// A Y4M file read into memory, for the tests of the reader.
#[cfg(test)]
#[derive(Debug)]
pub struct Video {
    pub width: usize,
    pub height: usize,
    pub chroma: ChromaSubsampling,
    pub bit_depth: BitDepth,
    /// Per frame: the Y, Cb and Cr planes.
    pub frames: Vec<[Vec<u8>; 3]>,
}

/// What the header line of a Y4M file says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub width: usize,
    pub height: usize,
    pub chroma: ChromaSubsampling,
    pub bit_depth: BitDepth,
    /// Where 4:2:0 chroma sits: left for `C420mpeg2` and `C420paldv`, whose
    /// chroma is co-sited with the left luma sample, otherwise the centre.
    pub siting: ChromaSiting,
    /// `XCOLORRANGE=FULL`, or video range otherwise.
    pub range: YcbcrRange,
    /// Numerator and denominator of the `F` field, or 0/0 without one.
    pub frame_rate: (u32, u32),
    /// Bytes of the Y, Cb and Cr planes of a frame.
    sizes: [usize; 3],
}

/// Reads an 8-bit or 10-bit 4:2:0 or 4:4:4 Y4M stream one frame at a time, so
/// that only one frame is in memory.
pub struct Reader<R> {
    input: R,
    header: Header,
}

impl<R: BufRead> Reader<R> {
    /// Reads the header line.
    pub fn new(mut input: R) -> Result<Self> {
        let mut line = Vec::new();
        (&mut input).take(MAX_LINE).read_until(b'\n', &mut line)?;
        if line.last() != Some(&b'\n') {
            return Err("not a Y4M file: no header line".into());
        }
        let header = parse_header(std::str::from_utf8(&line[..line.len() - 1])?)?;
        Ok(Self { input, header })
    }

    /// What the header line says.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The Y, Cb and Cr planes of the next frame, or `None` at the end.
    pub fn next_frame(&mut self) -> Result<Option<[Vec<u8>; 3]>> {
        let mut marker = Vec::new();
        if (&mut self.input)
            .take(MAX_LINE)
            .read_until(b'\n', &mut marker)?
            == 0
        {
            return Ok(None);
        }
        if marker.last() != Some(&b'\n') {
            return Err("truncated frame marker".into());
        }
        if !marker.starts_with(b"FRAME") {
            return Err("expected a FRAME marker".into());
        }
        let mut planes: [Vec<u8>; 3] = Default::default();
        for (plane, size) in planes.iter_mut().zip(self.header.sizes) {
            // Grows with the data that is there, so that a header naming an
            // absurd size gives an error rather than a failed allocation.
            let wanted = u64::try_from(size)?;
            (&mut self.input).take(wanted).read_to_end(plane)?;
            if plane.len() != size {
                return Err("truncated frame data".into());
            }
        }
        Ok(Some(planes))
    }
}

/// Opens a Y4M file to read frame by frame.
pub fn open(path: &Path) -> Result<Reader<BufReader<File>>> {
    File::open(path)
        .map_err(Into::into)
        .and_then(|file| Reader::new(BufReader::new(file)))
        .map_err(|error| format!("{}: {error}", path.display()).into())
}

/// Parses the contents of an 8-bit or 10-bit 4:2:0 or 4:4:4 Y4M file, for the
/// tests of the reader; the commands read frame by frame.
#[cfg(test)]
fn parse(data: &[u8]) -> Result<Video> {
    let mut reader = Reader::new(data)?;
    let mut frames = Vec::new();
    while let Some(frame) = reader.next_frame()? {
        frames.push(frame);
    }
    let header = *reader.header();
    Ok(Video {
        width: header.width,
        height: header.height,
        chroma: header.chroma,
        bit_depth: header.bit_depth,
        frames,
    })
}

/// Parses the header line, without its newline.
fn parse_header(line: &str) -> Result<Header> {
    if !line.starts_with("YUV4MPEG2") {
        return Err("not a Y4M file".into());
    }

    let (mut width, mut height, mut chroma, mut bit_depth) =
        (0usize, 0usize, ChromaSubsampling::Yuv420, BitDepth::Eight);
    let (mut siting, mut range, mut frame_rate) =
        (ChromaSiting::Center, YcbcrRange::Limited, (0, 0));
    for token in line.split_whitespace().skip(1) {
        // The first character names the field; a token that does not start
        // with a one-byte character is none of the fields we know.
        let Some((tag, value)) = token.split_at_checked(1) else {
            continue;
        };
        match tag {
            "W" => width = value.parse()?,
            "H" => height = value.parse()?,
            "F" => {
                frame_rate = value
                    .split_once(':')
                    .and_then(|(num, den)| Some((num.parse().ok()?, den.parse().ok()?)))
                    .ok_or_else(|| format!("unreadable frame rate F{value}"))?;
            }
            "X" if value == "COLORRANGE=FULL" => range = YcbcrRange::Full,
            "C" => {
                siting = match value {
                    "420mpeg2" | "420paldv" => ChromaSiting::Left,
                    _ => ChromaSiting::Center,
                };
                (chroma, bit_depth) = match value {
                    "420" | "420jpeg" | "420mpeg2" | "420paldv" => {
                        (ChromaSubsampling::Yuv420, BitDepth::Eight)
                    }
                    "444" => (ChromaSubsampling::Yuv444, BitDepth::Eight),
                    "420p10" => (ChromaSubsampling::Yuv420, BitDepth::Ten),
                    "444p10" => (ChromaSubsampling::Yuv444, BitDepth::Ten),
                    other => {
                        return Err(format!(
                            "unsupported Y4M colour space C{other} (8-bit or 10-bit 420 and 444 only)"
                        )
                        .into());
                    }
                }
            }
            _ => {}
        }
    }
    if width == 0 || height == 0 {
        return Err("the Y4M header has no picture size".into());
    }

    let (chroma_width, chroma_height) = match chroma {
        ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        ChromaSubsampling::Yuv444 => (width, height),
    };
    let bytes_per_sample = bit_depth.bytes_per_sample();
    // A header can name sizes whose product does not fit: that is an error,
    // not a picture of no samples.
    let plane_size = |w: usize, h: usize| {
        w.checked_mul(h)
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or("the picture size in the Y4M header is too large")
    };
    let sizes = [
        plane_size(width, height)?,
        plane_size(chroma_width, chroma_height)?,
        plane_size(chroma_width, chroma_height)?,
    ];

    Ok(Header {
        width,
        height,
        chroma,
        bit_depth,
        siting,
        range,
        frame_rate,
        sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(chroma: ChromaSubsampling, siting: ChromaSiting) -> StreamInfo {
        StreamInfo {
            width: 4,
            height: 2,
            chroma,
            siting,
            frame_rate: (60, 1),
            range: YcbcrRange::Limited,
            bit_depth: BitDepth::Eight,
        }
    }

    #[test]
    fn the_header_names_the_range_and_the_chroma_siting() {
        assert_eq!(
            info(ChromaSubsampling::Yuv420, ChromaSiting::Center).header_line(),
            "YUV4MPEG2 W4 H2 F60:1 Ip A1:1 XCOLORRANGE=LIMITED C420jpeg"
        );
        assert!(
            info(ChromaSubsampling::Yuv420, ChromaSiting::Left)
                .header_line()
                .ends_with(" C420mpeg2")
        );
        let full = StreamInfo {
            range: YcbcrRange::Full,
            ..info(ChromaSubsampling::Yuv444, ChromaSiting::Left)
        };
        assert!(full.header_line().ends_with("XCOLORRANGE=FULL C444"));
    }

    #[test]
    fn a_stream_without_a_frame_rate_gets_the_default_one() {
        for frame_rate in [(0, 0), (0, 1), (30, 0)] {
            let stream = StreamInfo {
                frame_rate,
                ..info(ChromaSubsampling::Yuv420, ChromaSiting::Center)
            };
            assert!(!stream.has_frame_rate(), "{frame_rate:?}");
            assert!(stream.header_line().contains(" F30:1 "), "{frame_rate:?}");
        }
    }

    /// A file with the given header and frames of `4 + 2 + 2` samples.
    fn file(header: &str, frames: usize) -> Vec<u8> {
        let mut data = format!("{header}\n").into_bytes();
        for frame in 0..frames {
            data.extend_from_slice(b"FRAME\n");
            data.extend((0..8).map(|i| (frame * 8 + i) as u8));
        }
        data
    }

    #[test]
    fn a_written_header_reads_back() {
        let stream = StreamInfo {
            height: 1,
            ..info(ChromaSubsampling::Yuv420, ChromaSiting::Left)
        };
        // 4x1 in 4:2:0: four luma samples, two chroma samples in each plane.
        let video = parse(&file(&stream.header_line(), 2)).unwrap();
        assert_eq!((video.width, video.height), (4, 1));
        assert_eq!(video.chroma, ChromaSubsampling::Yuv420);
        assert_eq!(video.frames.len(), 2);
        assert_eq!(video.frames[1][0], [8, 9, 10, 11]);
        assert_eq!(video.frames[1][1], [12, 13]);
        assert_eq!(video.frames[1][2], [14, 15]);
        // What the header signals comes back too.
        let data = file(&stream.header_line(), 2);
        let header = *Reader::new(data.as_slice()).unwrap().header();
        assert_eq!(header.siting, ChromaSiting::Left);
        assert_eq!(header.range, YcbcrRange::Limited);
        assert_eq!(header.frame_rate, stream.frame_rate);

        let full = StreamInfo {
            range: YcbcrRange::Full,
            ..stream
        };
        let data = file(&full.header_line(), 1);
        assert_eq!(
            Reader::new(data.as_slice()).unwrap().header().range,
            YcbcrRange::Full
        );
        assert!(parse(b"YUV4MPEG2 W4 H1 F30 C444\n").is_err());
        let paldv = b"YUV4MPEG2 W4 H2 C420paldv\n";
        assert_eq!(
            Reader::new(&paldv[..]).unwrap().header().siting,
            ChromaSiting::Left
        );
    }

    #[test]
    fn sizes_that_overflow_are_an_error_not_an_empty_picture() {
        for header in [
            "YUV4MPEG2 W4294967296 H4294967296 C444",
            "YUV4MPEG2 W18446744073709551615 H2 C444",
            "YUV4MPEG2 W18446744073709551615 H1 C444",
        ] {
            assert!(parse(&file(header, 1)).is_err(), "{header}");
        }
    }

    #[test]
    fn damaged_files_are_errors() {
        assert!(parse(b"").is_err());
        assert!(parse(b"not a y4m file\n").is_err());
        assert!(parse(b"YUV4MPEG2 W4 H2 C422\n").is_err());
        assert!(parse(b"YUV4MPEG2 W0 H2 C444\n").is_err());
        // A frame that stops short, and something other than a frame marker.
        let mut cut = file("YUV4MPEG2 W4 H1 C420", 1);
        cut.pop();
        assert!(parse(&cut).is_err());
        assert!(parse(b"YUV4MPEG2 W4 H1 C420\nJUNK\n").is_err());
    }

    #[test]
    fn header_tokens_that_do_not_start_with_ascii_are_skipped() {
        // Splitting such a token after its first byte would panic.
        let video = parse(&file("YUV4MPEG2 W4 H1 한국어 C420", 1)).unwrap();
        assert_eq!((video.width, video.height), (4, 1));
        assert!(parse(&file("YUV4MPEG2 W", 1)).is_err());
    }

    #[test]
    fn ten_bit_headers_and_sample_buffers_round_trip() {
        for (chroma, expected_tag, chroma_samples) in [
            (ChromaSubsampling::Yuv420, "C420p10", 2usize),
            (ChromaSubsampling::Yuv444, "C444p10", 8usize),
        ] {
            let stream = StreamInfo {
                bit_depth: BitDepth::Ten,
                ..info(chroma, ChromaSiting::Center)
            };
            let line = stream.header_line();
            assert!(line.ends_with(expected_tag), "{line}");

            let mut data = format!("{line}\nFRAME\n").into_bytes();
            let y_samples = [0u16, 64, 512, 940, 1023, 1, 255, 768];
            for s in y_samples {
                data.extend_from_slice(&s.to_le_bytes());
            }
            for c in 0..chroma_samples * 2 {
                data.extend_from_slice(&((128 + c as u16 * 111) % 1024).to_le_bytes());
            }

            let header = *Reader::new(data.as_slice()).unwrap().header();
            assert_eq!(header.siting, ChromaSiting::Center);
            assert_eq!(header.bit_depth, BitDepth::Ten);

            let video = parse(&data).unwrap();
            assert_eq!(video.bit_depth, BitDepth::Ten);
            assert_eq!(video.chroma, chroma);
            assert_eq!(video.frames.len(), 1);
            assert_eq!(decode_u10_le_plane(&video.frames[0][0]).unwrap(), y_samples);
        }
    }

    #[test]
    fn ten_bit_sample_conversions_preserve_all_10bit_levels_and_reject_bad_inputs() {
        for code in 0u16..=1023 {
            let unorm = u10_to_unorm16(code).unwrap();
            assert_eq!(unorm16_to_u10(unorm), code, "code {code}");
        }
        // Exhaustively verify unorm16_to_u10 matches exact rounding over all u16.
        for sample in 0u16..=u16::MAX {
            let exact = (f64::from(sample) * 1023.0 / 65_535.0).round_ties_even() as u16;
            assert_eq!(unorm16_to_u10(sample), exact, "sample {sample}");
        }
        assert_eq!(unorm16_to_u10(19_763), 309);

        // Limited-range cross-depth conversion preserves nominal BT.709/BT.2020 levels.
        for (c8, c10) in [(16u8, 64u16), (128, 512), (235, 940), (240, 960)] {
            let u16_from_8 = unorm16(f32::from(c8) / 255.0);
            assert_eq!(
                unorm16_to_u10_with_range(u16_from_8, 8, YcbcrRange::Limited),
                c10,
                "8->10 limited c8={c8}"
            );
            let u16_from_10 = u10_to_unorm16(c10).unwrap();
            assert_eq!(
                unorm16_to_u8_limited_from_u10(u16_from_10),
                c8,
                "10->8 limited c10={c10}"
            );
            let u8_from_10 = ((f32::from(c10) / 1023.0) * 255.0).round_ties_even() as u8;
            assert_eq!(
                unorm8_to_u8_limited_from_u10(u8_from_10),
                c8,
                "10->8 fallback c10={c10}"
            );
        }
        assert_eq!(unorm16_to_u10_with_range(65_535, 8, YcbcrRange::Full), 1023);
        for (c16, c10) in [
            (4096u16, 64u16),
            (32_768, 512),
            (60_160, 940),
            (61_440, 960),
        ] {
            assert_eq!(
                unorm16_to_u10_with_range(c16, 16, YcbcrRange::Limited),
                c10,
                "16->10 limited c16={c16}"
            );
        }
        assert!(u10_to_unorm16(1024).is_err());
        assert!(decode_u10_le_plane(&[0, 4]).is_err()); // 1024 in LE
        assert!(decode_u10_le_plane(&[0]).is_err()); // odd byte count
        assert!(u10_plane_to_unorm16(&[0, 4]).is_err());
        assert!(u10_plane_to_unorm16(&[0]).is_err());
    }
}
