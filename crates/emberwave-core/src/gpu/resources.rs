//! Buffers, bind groups and read-backs: what the decoder and the encoder share.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use super::context::{QueueTurns, submit, take_turn};
use super::error::GpuError;
use super::pipelines::{Stage, with_error_scope};

/// Maximum number of upload ring slots a decoder keeps for pipelined
/// non-blocking submissions. Slot 0 is created upfront; extra slots are added
/// lazily only when earlier slots are still in flight on the GPU.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) const MAX_UPLOAD_RING_SLOTS: usize = 3;

const MAP_PENDING: u8 = 0;
const MAP_ERR_BIT: u8 = 0x80;

/// Mapping state of one upload slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotMapState {
    Mapped,
    Pending { expected: u8 },
    Unmapped,
}

/// Zero-allocation tracker for asynchronous `MAP_WRITE` buffer remapping.
///
/// Reuses one `Arc<AtomicU8>` across the lifetime of an upload slot so steady-
/// state frames neither allocate channels nor block when an earlier slot is
/// ready.
#[derive(Debug)]
pub(crate) struct MapTracker {
    state: SlotMapState,
    flag: Arc<AtomicU8>,
}

impl MapTracker {
    pub(crate) fn mapped() -> Self {
        Self {
            state: SlotMapState::Mapped,
            flag: Arc::new(AtomicU8::new(0)),
        }
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.state == SlotMapState::Mapped
    }

    pub(crate) fn is_pending(&self) -> bool {
        matches!(self.state, SlotMapState::Pending { .. })
    }

    pub(crate) fn mark_mapped(&mut self) {
        self.state = SlotMapState::Mapped;
    }

    pub(crate) fn mark_unmapped(&mut self) {
        self.state = SlotMapState::Unmapped;
    }

    /// Checks whether a pending `map_async` has completed without polling.
    pub(crate) fn poll_ready(&mut self) -> Result<bool, GpuError> {
        match self.state {
            SlotMapState::Mapped => Ok(true),
            SlotMapState::Unmapped => Ok(false),
            SlotMapState::Pending { expected } => {
                let value = self.flag.load(Ordering::Acquire);
                if value & MAP_ERR_BIT != 0 {
                    self.state = SlotMapState::Unmapped;
                    Err(GpuError::Device(
                        "mapping the upload buffer failed".to_owned(),
                    ))
                } else if value >= expected {
                    self.state = SlotMapState::Mapped;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Schedules asynchronous write mapping for `buffers` after a submission.
    pub(crate) fn begin_remap(&mut self, buffers: &[&wgpu::Buffer]) {
        self.flag.store(MAP_PENDING, Ordering::Release);
        for buffer in buffers {
            let flag = Arc::clone(&self.flag);
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Write, move |result| match result {
                    Ok(()) => {
                        flag.fetch_add(1, Ordering::Release);
                    }
                    Err(_) => {
                        flag.fetch_or(MAP_ERR_BIT, Ordering::Release);
                    }
                });
        }
        self.state = SlotMapState::Pending {
            expected: buffers.len() as u8,
        };
    }

    /// Ensures `buffers` are mapped for writing, waiting only if still in flight.
    pub(crate) fn ensure_mapped(
        &mut self,
        device: &wgpu::Device,
        buffers: &[&wgpu::Buffer],
    ) -> Result<(), GpuError> {
        if self.poll_ready()? {
            return Ok(());
        }
        if self.state == SlotMapState::Unmapped {
            with_error_scope(device, || {
                self.begin_remap(buffers);
            })?;
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                GpuError::Device(format!("mapping the upload buffer failed: {error}"))
            })?;
        if self.poll_ready()? {
            Ok(())
        } else {
            self.state = SlotMapState::Unmapped;
            Err(GpuError::Device(
                "mapping the upload buffer failed".to_owned(),
            ))
        }
    }
}

/// Shared dequantisation bindings needed when constructing or resizing upload slots.
#[derive(Clone, Copy)]
pub(crate) struct DequantTarget<'a> {
    pub stage: &'a Stage,
    pub pyramid: &'a wgpu::Buffer,
    pub params: &'a wgpu::Buffer,
    pub dirty_ballots: &'a wgpu::Buffer,
}

/// Shared concealment bindings needed when building per-slot concealment bind groups.
#[derive(Clone, Copy)]
pub(crate) struct ConcealTarget<'a> {
    pub layout: &'a wgpu::BindGroupLayout,
    pub pyramid: &'a wgpu::Buffer,
    pub dequant_params: &'a wgpu::Buffer,
    pub uniform: &'a wgpu::Buffer,
    pub last: &'a wgpu::Buffer,
}

/// Framed byte slices and sizing bounds for mapped upload paths.
#[derive(Clone, Copy)]
pub(crate) struct FrameUpload<'a> {
    pub table: &'a [u8],
    pub work: &'a [u8],
    pub payload: &'a [u8],
    pub table_capacity_bytes: u64,
    pub work_capacity_bytes: u64,
    pub needed_words: usize,
    pub min_payload_words: usize,
}

/// One UMA upload slot whose primary buffers are directly writable by the CPU
/// and bound directly by the dequantisation shader.
pub(crate) struct UmaSlot {
    pub block_offsets: wgpu::Buffer,
    pub dequant_work: wgpu::Buffer,
    pub payload: wgpu::Buffer,
    pub payload_capacity_words: usize,
    pub dequant_bind_group: wgpu::BindGroup,
    pub conceal_bind_group: Option<wgpu::BindGroup>,
    pub tracker: MapTracker,
}

impl UmaSlot {
    pub(crate) fn new(
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        table_bytes: u64,
        work_bytes: u64,
        payload_capacity_words: usize,
        shared_uploads: bool,
    ) -> Self {
        let usage = frame_buffer_usage(shared_uploads);
        let block_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block offsets"),
            size: table_bytes.max(16),
            usage,
            mapped_at_creation: shared_uploads,
        });
        let dequant_work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dequant work"),
            size: work_bytes.max(16),
            usage,
            mapped_at_creation: shared_uploads,
        });
        let payload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("payload"),
            size: (payload_capacity_words as u64 * 4).max(16),
            usage,
            mapped_at_creation: shared_uploads,
        });
        let dequant_bind_group = dequant_bind_group(
            device,
            dequant.stage,
            &payload,
            &dequant_work,
            dequant.pyramid,
            dequant.params,
            dequant.dirty_ballots,
        );
        let tracker = if shared_uploads {
            MapTracker::mapped()
        } else {
            MapTracker::mapped_initial_unmapped()
        };
        Self {
            block_offsets,
            dequant_work,
            payload,
            payload_capacity_words,
            dequant_bind_group,
            conceal_bind_group: None,
            tracker,
        }
    }

    pub(crate) fn ensure_payload_capacity(
        &mut self,
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        needed_words: usize,
        min_payload_words: usize,
        shared_uploads: bool,
        map_new_buffer: bool,
    ) -> Result<(), GpuError> {
        if needed_words <= self.payload_capacity_words {
            return Ok(());
        }
        let limit_bytes = buffer_limit(device);
        let capacity_bytes = payload_capacity_bytes(needed_words, min_payload_words, limit_bytes)
            .ok_or(GpuError::LimitExceeded {
            what: "the payload of this frame",
            needed: needed_words as u64 * 4,
            limit: limit_bytes & !3,
        })?;
        // If `block_offsets` / `dequant_work` are currently mapped while `payload` is
        // recreated as `mapped_at_creation`, all remain mapped together.
        let usage = frame_buffer_usage(shared_uploads);
        let mapped_at_creation = shared_uploads && map_new_buffer;
        let (payload, bind_group) = with_error_scope(device, || {
            let payload = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("payload"),
                size: capacity_bytes,
                usage,
                mapped_at_creation,
            });
            let bind_group = dequant_bind_group(
                device,
                dequant.stage,
                &payload,
                &self.dequant_work,
                dequant.pyramid,
                dequant.params,
                dequant.dirty_ballots,
            );
            (payload, bind_group)
        })?;
        std::mem::replace(&mut self.payload, payload).destroy();
        self.payload_capacity_words = (capacity_bytes / 4) as usize;
        self.dequant_bind_group = bind_group;
        if shared_uploads && !mapped_at_creation && self.tracker.is_mapped() {
            // QueueWrite / unmapped caller expects an unmapped slot.
            self.block_offsets.unmap();
            self.dequant_work.unmap();
            self.tracker.mark_unmapped();
        }
        Ok(())
    }

    pub(crate) fn ensure_unmapped(&mut self, device: &wgpu::Device) -> Result<(), GpuError> {
        if self.tracker.is_pending() {
            self.tracker.ensure_mapped(
                device,
                &[&self.block_offsets, &self.dequant_work, &self.payload],
            )?;
        }
        if self.tracker.is_mapped() {
            with_error_scope(device, || {
                self.block_offsets.unmap();
                self.dequant_work.unmap();
                self.payload.unmap();
            })?;
            self.tracker.mark_unmapped();
        }
        Ok(())
    }
}

impl MapTracker {
    fn mapped_initial_unmapped() -> Self {
        Self {
            state: SlotMapState::Unmapped,
            flag: Arc::new(AtomicU8::new(0)),
        }
    }
}

/// One persistent staging slot (`MAP_WRITE | COPY_SRC`) for discrete GPUs.
pub(crate) struct StagingSlot {
    pub buffer: wgpu::Buffer,
    pub tracker: MapTracker,
}

impl StagingSlot {
    pub(crate) fn new(device: &wgpu::Device, capacity_bytes: u64) -> Result<Self, GpuError> {
        let buffer = with_error_scope(device, || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame upload"),
                size: capacity_bytes,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        })?;
        Ok(Self {
            buffer,
            tracker: MapTracker::mapped(),
        })
    }

    pub(crate) fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        capacity_bytes: u64,
    ) -> Result<(), GpuError> {
        if self.buffer.size() >= capacity_bytes {
            return Ok(());
        }
        let buffer = with_error_scope(device, || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame upload"),
                size: capacity_bytes,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        })?;
        std::mem::replace(&mut self.buffer, buffer).destroy();
        self.tracker.mark_mapped();
        Ok(())
    }
}

/// Manages frame upload buffers for both UMA shared memory and discrete staging rings.
pub(crate) enum DecoderUploads {
    Uma {
        slots: Vec<UmaSlot>,
        next_slot: usize,
    },
    Staging {
        slot: UmaSlot,
        staging: Vec<StagingSlot>,
        next_staging: usize,
    },
}

/// Result of preparing and writing a frame upload before GPU command encoding.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedUpload {
    pub slot_index: usize,
    pub table_bytes: u64,
    pub work_bytes: u64,
    pub payload_bytes: u64,
    pub needs_copy: bool,
}

impl DecoderUploads {
    pub(crate) fn new(
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        table_bytes: u64,
        work_bytes: u64,
        min_payload_words: usize,
        shared_uploads: bool,
    ) -> Self {
        let slot = UmaSlot::new(
            device,
            dequant,
            table_bytes,
            work_bytes,
            min_payload_words,
            shared_uploads,
        );
        if shared_uploads {
            Self::Uma {
                slots: vec![slot],
                next_slot: 0,
            }
        } else {
            Self::Staging {
                slot,
                staging: Vec::new(),
                next_staging: 0,
            }
        }
    }

    pub(crate) fn is_shared(&self) -> bool {
        matches!(self, Self::Uma { .. })
    }

    /// Primary slot (`0`) buffers for inspection or `Queue::write_buffer`.
    pub(crate) fn primary_slot(&self) -> &UmaSlot {
        match self {
            Self::Uma { slots, .. } => &slots[0],
            Self::Staging { slot, .. } => slot,
        }
    }

    pub(crate) fn slot(&self, index: usize) -> &UmaSlot {
        match self {
            Self::Uma { slots, .. } => &slots[index],
            Self::Staging { slot, .. } => slot,
        }
    }

    pub(crate) fn staging_buffer(&self, index: usize) -> Option<&wgpu::Buffer> {
        match self {
            Self::Uma { .. } => None,
            Self::Staging { staging, .. } => staging.get(index).map(|s| &s.buffer),
        }
    }

    /// Ensures the shader-read buffer can hold `needed_words` for `Queue::write_buffer`
    /// paths (`record_assembled` and `TimedUpload::QueueWrite`) and unmaps slot 0.
    pub(crate) fn prepare_queue_write(
        &mut self,
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        needed_words: usize,
        min_payload_words: usize,
    ) -> Result<(), GpuError> {
        match self {
            Self::Uma { slots, .. } => {
                slots[0].ensure_unmapped(device)?;
                slots[0].ensure_payload_capacity(
                    device,
                    dequant,
                    needed_words,
                    min_payload_words,
                    true,
                    false,
                )
            }
            Self::Staging { slot, .. } => slot.ensure_payload_capacity(
                device,
                dequant,
                needed_words,
                min_payload_words,
                false,
                false,
            ),
        }
    }

    /// Writes `upload` directly into a mapped slot (UMA) or staging buffer
    /// (discrete) without touching the queue.
    pub(crate) fn write_mapped(
        &mut self,
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        upload: FrameUpload<'_>,
        allow_ring: bool,
    ) -> Result<PreparedUpload, GpuError> {
        match self {
            Self::Uma { slots, next_slot } => {
                let slot_index =
                    acquire_uma_slot(slots, next_slot, device, dequant, &upload, allow_ring)?;
                let slot = &mut slots[slot_index];
                if !slot.tracker.is_mapped() {
                    slot.tracker.ensure_mapped(
                        device,
                        &[&slot.block_offsets, &slot.dequant_work, &slot.payload],
                    )?;
                }
                slot.ensure_payload_capacity(
                    device,
                    dequant,
                    upload.needed_words,
                    upload.min_payload_words,
                    true,
                    true,
                )?;
                let written = write_uma_slot(slot, upload.table, upload.work, upload.payload);
                let unmapped = with_error_scope(device, || {
                    slot.block_offsets.unmap();
                    slot.dequant_work.unmap();
                    slot.payload.unmap();
                });
                slot.tracker.mark_unmapped();
                written?;
                unmapped?;
                Ok(PreparedUpload {
                    slot_index,
                    table_bytes: 0,
                    work_bytes: 0,
                    payload_bytes: 0,
                    needs_copy: false,
                })
            }
            Self::Staging {
                slot,
                staging,
                next_staging,
            } => {
                slot.ensure_payload_capacity(
                    device,
                    dequant,
                    upload.needed_words,
                    upload.min_payload_words,
                    false,
                    false,
                )?;
                let capacity_bytes = upload.table_capacity_bytes
                    + upload.work_capacity_bytes
                    + slot.payload_capacity_words as u64 * 4;
                let slot_index = acquire_staging_slot(
                    staging,
                    next_staging,
                    device,
                    capacity_bytes,
                    allow_ring,
                )?;
                let entry = &mut staging[slot_index];
                if !entry.tracker.is_mapped() {
                    entry.tracker.ensure_mapped(device, &[&entry.buffer])?;
                }
                entry.ensure_capacity(device, capacity_bytes)?;
                let written = write_staging_slot(
                    entry,
                    upload.table_capacity_bytes,
                    upload.work_capacity_bytes,
                    upload.table,
                    upload.work,
                    upload.payload,
                );
                let unmapped = with_error_scope(device, || entry.buffer.unmap());
                entry.tracker.mark_unmapped();
                written?;
                unmapped?;
                Ok(PreparedUpload {
                    slot_index,
                    table_bytes: upload.table.len() as u64,
                    work_bytes: upload.work.len() as u64,
                    payload_bytes: upload.payload.len() as u64,
                    needs_copy: true,
                })
            }
        }
    }

    /// Attempts to write `upload` directly into UMA slot 0 for `record_assembled`
    /// when it is already mapped, returning `Ok(true)` if written in place or
    /// `Ok(false)` if the caller should use `queue.write_buffer`.
    pub(crate) fn try_write_uma_for_record(
        &mut self,
        device: &wgpu::Device,
        dequant: DequantTarget<'_>,
        upload: FrameUpload<'_>,
    ) -> Result<bool, GpuError> {
        let Self::Uma { slots, .. } = self else {
            return Ok(false);
        };
        let slot = &mut slots[0];
        if slot.tracker.is_pending() {
            let _ = device.poll(wgpu::PollType::Poll);
            let _ = slot.tracker.poll_ready()?;
        }
        if !slot.tracker.is_mapped() {
            return Ok(false);
        }
        slot.ensure_payload_capacity(
            device,
            dequant,
            upload.needed_words,
            upload.min_payload_words,
            true,
            true,
        )?;
        let written = write_uma_slot(slot, upload.table, upload.work, upload.payload);
        let unmapped = with_error_scope(device, || {
            slot.block_offsets.unmap();
            slot.dequant_work.unmap();
            slot.payload.unmap();
        });
        slot.tracker.mark_unmapped();
        written?;
        unmapped?;
        Ok(true)
    }

    /// Schedules asynchronous remapping of `slot_index` right after its GPU
    /// submission so future frames find it mapped without waiting.
    pub(crate) fn begin_remap(&mut self, slot_index: usize) {
        match self {
            Self::Uma { slots, .. } => {
                let slot = &mut slots[slot_index];
                slot.tracker
                    .begin_remap(&[&slot.block_offsets, &slot.dequant_work, &slot.payload]);
            }
            Self::Staging { staging, .. } => {
                let entry = &mut staging[slot_index];
                entry.tracker.begin_remap(&[&entry.buffer]);
            }
        }
    }

    /// Ensures the concealment bind group exists for `slot_index` using `conceal`.
    pub(crate) fn ensure_conceal_bind_group(
        &mut self,
        slot_index: usize,
        device: &wgpu::Device,
        conceal: ConcealTarget<'_>,
    ) {
        let slot = match self {
            Self::Uma { slots, .. } => &mut slots[slot_index],
            Self::Staging { slot, .. } => slot,
        };
        if slot.conceal_bind_group.is_none() {
            slot.conceal_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("conceal"),
                layout: conceal.layout,
                entries: &[
                    entry(0, &slot.block_offsets),
                    entry(1, conceal.pyramid),
                    entry(2, conceal.dequant_params),
                    entry(3, conceal.uniform),
                    entry(4, conceal.last),
                ],
            }));
        }
    }

    pub(crate) fn conceal_bind_group(&self, slot_index: usize) -> Option<&wgpu::BindGroup> {
        self.slot(slot_index).conceal_bind_group.as_ref()
    }

    /// Destroys all owned upload buffers.
    pub(crate) fn destroy_all(&self) {
        match self {
            Self::Uma { slots, .. } => {
                for slot in slots {
                    slot.block_offsets.destroy();
                    slot.dequant_work.destroy();
                    slot.payload.destroy();
                }
            }
            Self::Staging { slot, staging, .. } => {
                slot.block_offsets.destroy();
                slot.dequant_work.destroy();
                slot.payload.destroy();
                for entry in staging {
                    entry.buffer.destroy();
                }
            }
        }
    }
}

fn acquire_uma_slot(
    slots: &mut Vec<UmaSlot>,
    next_slot: &mut usize,
    device: &wgpu::Device,
    dequant: DequantTarget<'_>,
    upload: &FrameUpload<'_>,
    allow_ring: bool,
) -> Result<usize, GpuError> {
    if !allow_ring {
        if slots[0].tracker.is_pending() {
            let _ = device.poll(wgpu::PollType::Poll);
            let _ = slots[0].tracker.poll_ready()?;
        }
        *next_slot = 0;
        return Ok(0);
    }

    if slots.iter().any(|slot| slot.tracker.is_pending()) {
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in slots.iter_mut() {
            if slot.tracker.is_pending() {
                let _ = slot.tracker.poll_ready()?;
            }
        }
    }

    let len = slots.len();
    for offset in 0..len {
        let idx = (*next_slot + offset) % len;
        if slots[idx].tracker.is_mapped() {
            *next_slot = (idx + 1) % len;
            return Ok(idx);
        }
    }

    if len < MAX_UPLOAD_RING_SLOTS {
        let new_slot = with_error_scope(device, || {
            UmaSlot::new(
                device,
                dequant,
                upload.table_capacity_bytes,
                upload.work_capacity_bytes,
                upload.min_payload_words,
                true,
            )
        })?;
        let idx = slots.len();
        slots.push(new_slot);
        *next_slot = (idx + 1) % slots.len();
        return Ok(idx);
    }

    let idx = *next_slot % len;
    *next_slot = (idx + 1) % len;
    Ok(idx)
}

fn acquire_staging_slot(
    staging: &mut Vec<StagingSlot>,
    next_staging: &mut usize,
    device: &wgpu::Device,
    capacity_bytes: u64,
    allow_ring: bool,
) -> Result<usize, GpuError> {
    if staging.is_empty() {
        staging.push(StagingSlot::new(device, capacity_bytes)?);
        *next_staging = 0;
        return Ok(0);
    }

    if !allow_ring {
        if staging[0].tracker.is_pending() {
            let _ = device.poll(wgpu::PollType::Poll);
            let _ = staging[0].tracker.poll_ready()?;
        }
        *next_staging = 0;
        return Ok(0);
    }

    if staging.iter().any(|slot| slot.tracker.is_pending()) {
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in staging.iter_mut() {
            if slot.tracker.is_pending() {
                let _ = slot.tracker.poll_ready()?;
            }
        }
    }

    let len = staging.len();
    for offset in 0..len {
        let idx = (*next_staging + offset) % len;
        if staging[idx].tracker.is_mapped() {
            *next_staging = (idx + 1) % len;
            return Ok(idx);
        }
    }

    if len < MAX_UPLOAD_RING_SLOTS {
        let new_slot = StagingSlot::new(device, capacity_bytes)?;
        let idx = staging.len();
        staging.push(new_slot);
        *next_staging = (idx + 1) % staging.len();
        return Ok(idx);
    }

    let idx = *next_staging % len;
    *next_staging = (idx + 1) % len;
    Ok(idx)
}

fn write_uma_slot(
    slot: &UmaSlot,
    table: &[u8],
    work: &[u8],
    payload: &[u8],
) -> Result<(), GpuError> {
    if !table.is_empty() {
        let mut table_view = slot
            .block_offsets
            .slice(..table.len() as u64)
            .get_mapped_range_mut()
            .map_err(|error| GpuError::Device(error.to_string()))?;
        table_view.slice(..).copy_from_slice(table);
    }

    let mut work_view = slot
        .dequant_work
        .slice(..work.len() as u64)
        .get_mapped_range_mut()
        .map_err(|error| GpuError::Device(error.to_string()))?;
    work_view.slice(..).copy_from_slice(work);
    drop(work_view);

    let mut payload_view = slot
        .payload
        .slice(..payload.len() as u64)
        .get_mapped_range_mut()
        .map_err(|error| GpuError::Device(error.to_string()))?;
    payload_view.slice(..).copy_from_slice(payload);
    Ok(())
}

fn write_staging_slot(
    slot: &StagingSlot,
    work_start: u64,
    work_capacity: u64,
    table: &[u8],
    work: &[u8],
    payload: &[u8],
) -> Result<(), GpuError> {
    let payload_start = work_start + work_capacity;
    let total_bytes = payload_start + payload.len() as u64;
    let mut view = slot
        .buffer
        .slice(..total_bytes)
        .get_mapped_range_mut()
        .map_err(|error| GpuError::Device(error.to_string()))?;
    if !table.is_empty() {
        view.slice(..table.len()).copy_from_slice(table);
    }
    let w_start = work_start as usize;
    view.slice(w_start..w_start + work.len())
        .copy_from_slice(work);
    let start = payload_start as usize;
    view.slice(start..start + payload.len())
        .copy_from_slice(payload);
    Ok(())
}

/// The size in bytes of a payload buffer that holds `needed_words` words on a
/// device whose buffers can be at most `limit_bytes` large, or `None` if even
/// the exact size is too much.
///
/// The size is rounded up to a power of two, so that a stream of slowly
/// growing frames does not reallocate every time, but never beyond the limit.
pub(crate) fn payload_capacity_bytes(
    needed_words: usize,
    min_payload_words: usize,
    limit_bytes: u64,
) -> Option<u64> {
    let limit = limit_bytes & !3;
    let needed = needed_words as u64 * 4;
    if needed > limit {
        return None;
    }
    let rounded = needed_words.next_power_of_two().max(min_payload_words) as u64 * 4;
    Some(rounded.min(limit))
}

/// The usage of the buffers the shaders read a frame from: written through
/// the queue, or with `shared_uploads` also mapped by the CPU (decision D53).
pub(crate) fn frame_buffer_usage(shared_uploads: bool) -> wgpu::BufferUsages {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    usage.set(wgpu::BufferUsages::MAP_WRITE, shared_uploads);
    usage
}

pub(crate) fn dequant_bind_group(
    device: &wgpu::Device,
    stage: &Stage,
    payload: &wgpu::Buffer,
    dequant_work: &wgpu::Buffer,
    pyramid: &wgpu::Buffer,
    params: &wgpu::Buffer,
    dirty_ballots: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dequant"),
        layout: &stage.layout,
        entries: &[
            entry(0, payload),
            entry(1, dequant_work),
            entry(2, pyramid),
            entry(3, params),
            entry(4, dirty_ballots),
        ],
    })
}

/// The most bytes one buffer can have and still be bound as a storage buffer.
pub(crate) fn buffer_limit(device: &wgpu::Device) -> u64 {
    let limits = device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}

/// An error unless a buffer of `bytes` bytes fits the device's limit;
/// `what` names it in the error.
pub(crate) fn check_size(
    device: &wgpu::Device,
    what: &'static str,
    bytes: u64,
) -> Result<(), GpuError> {
    let limit = buffer_limit(device);
    if bytes > limit {
        Err(GpuError::LimitExceeded {
            what,
            needed: bytes,
            limit,
        })
    } else {
        Ok(())
    }
}

/// A buffer of at least 16 bytes, uninitialised.
pub(crate) fn buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage,
        mapped_at_creation: false,
    })
}

/// A bind group entry for a whole buffer.
pub(crate) fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A bind group entry for a texture view.
pub(crate) fn texture(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// A uniform buffer holding `value`, written through the queue. Not
/// `DeviceExt::create_buffer_init`, which maps the buffer at its creation
/// and panics when the device is lost (issue #35).
pub(crate) fn uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
    buffer
}

/// Maps `buffers` for writing and waits until they are. Callers map buffers
/// the GPU is done with, so this waits for nothing but wgpu's bookkeeping;
/// only a lost device makes it fail.
pub(crate) fn map_for_writing(
    device: &wgpu::Device,
    buffers: &[&wgpu::Buffer],
) -> Result<(), GpuError> {
    let mut tracker = MapTracker::mapped_initial_unmapped();
    tracker.ensure_mapped(device, buffers)
}

/// Waits for the GPU and copies a mappable buffer's first `bytes` bytes out.
pub(crate) fn read_back(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u8>, GpuError> {
    read_back_with(device, buffer, bytes, <[u8]>::to_vec)
}

/// Waits for the GPU, maps a mappable buffer's first `bytes` bytes and hands
/// them to `consume`, which can take what it needs without a copy of the
/// whole. The buffer is unmapped afterwards, also when `consume` panics.
pub(crate) fn read_back_with<T>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    bytes: u64,
    consume: impl FnOnce(&[u8]) -> T,
) -> Result<T, GpuError> {
    // Mapping and unmapping report errors through the error scopes like the
    // other calls. Waiting cannot: wgpu treats a device error while polling as
    // fatal (decision D22).
    with_error_scope(device, || {
        let slice = buffer.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver is only gone if this function already returned.
            let _ = sender.send(result);
        });

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuError::Readback(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| GpuError::Readback(error.to_string()))?
            .map_err(|error| GpuError::Readback(error.to_string()))?;

        let view = slice
            .get_mapped_range()
            .map_err(|error| GpuError::Readback(error.to_string()))?;
        let mapped = Mapped {
            view: Some(view),
            buffer,
        };
        Ok(consume(mapped.bytes()))
    })?
}

/// A mapped range of a buffer. Dropping it, in the error scope of the read
/// and during a panic too, lets go of the range and then unmaps the buffer,
/// which must not be unmapped while it is not mapped: that is an error.
struct Mapped<'a> {
    view: Option<wgpu::BufferView>,
    buffer: &'a wgpu::Buffer,
}

impl Mapped<'_> {
    fn bytes(&self) -> &[u8] {
        self.view.as_deref().unwrap_or_default()
    }
}

impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        self.view = None;
        self.buffer.unmap();
    }
}

/// Timestamp queries at the boundaries of compute passes: a beginning and an
/// end for each of a number of stages, a stage being one pass or several in
/// a row. Only on a device created with timestamps.
pub(crate) struct StageQueries {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    stages: u32,
}

impl StageQueries {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, stages: u32) -> Self {
        let bytes = u64::from(2 * stages) * 8;
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * stages,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: f64::from(queue.get_timestamp_period()),
            stages,
        }
    }

    /// The timestamps of a pass that begins stage `begins`, ends stage
    /// `ends`, or both.
    pub(crate) fn writes(
        &self,
        begins: Option<u32>,
        ends: Option<u32>,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: begins.map(|stage| 2 * stage),
            end_of_pass_write_index: ends.map(|stage| 2 * stage + 1),
        }
    }

    /// The time of each stage, once the work with the timestamps is done:
    /// `None` for a stage whose end came before its beginning, and `None`
    /// altogether if the device wrote no timestamps. A stage shorter than one
    /// tick of the timer is a valid measurement of zero, not a missing one.
    ///
    /// The timestamps are resolved here, in a submission of their own. With
    /// the work, in its command buffer, Metal now and then resolved the last
    /// stages before their timestamps were written, and gave the previous
    /// frame's or an end before the beginning. The submission takes a turn of
    /// `turns`, the context's, if it has any (decision D60).
    pub(crate) fn read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        turns: Option<&QueueTurns>,
    ) -> Result<Option<Vec<Option<Duration>>>, GpuError> {
        self.read_count(device, queue, turns, self.stages)
    }

    /// Reads the timestamps for the first `stages` stages.
    pub(crate) fn read_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        turns: Option<&QueueTurns>,
        stages: u32,
    ) -> Result<Option<Vec<Option<Duration>>>, GpuError> {
        let stages = stages.min(self.stages);
        let bytes = u64::from(2 * stages) * 8;
        let mut recording = take_turn(turns, device, queue);
        with_error_scope(device, || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamps"),
            });
            encoder.resolve_query_set(&self.query_set, 0..2 * stages, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, bytes);
            submit(queue, turns, [encoder.finish()]);
        })?;
        recording.submitted();
        drop(recording);
        let bytes = read_back(device, &self.readback, bytes)?;
        let ticks: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        // A device that wrote nothing leaves zeros.
        if ticks.iter().all(|&tick| tick == 0) {
            return Ok(None);
        }
        Ok(Some(
            ticks
                .chunks_exact(2)
                .map(|pair| {
                    (pair[1] >= pair[0]).then(|| {
                        Duration::from_secs_f64((pair[1] - pair[0]) as f64 * self.period_ns * 1e-9)
                    })
                })
                .collect(),
        ))
    }
}
