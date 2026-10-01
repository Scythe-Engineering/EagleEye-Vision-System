//! ABI 2 compatibility for the existing Python adapter and frozen-library comparisons.

use std::cell::UnsafeCell;
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::detection::Detection;
use crate::detector::{image_span, mapped_image, Detector, Settings};

#[repr(C)]
pub struct Config {
    abi_version: u32,
    struct_size: u32,
    families: *const c_char,
    nthreads: i32,
    refine_edges: i32,
    quad_decimate: f64,
    quad_sigma: f64,
    decode_sharpening: f64,
}

pub struct Handle {
    busy: AtomicBool,
    core: UnsafeCell<Detector>,
}

struct BusyGuard<'a>(&'a AtomicBool);

impl Drop for BusyGuard<'_> {
    /// Release exclusive access even when detection returns an error or panics.
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Copy bounded diagnostic bytes to a caller-owned, NUL-terminated buffer.
///
/// # Safety
/// A non-null buffer must hold at least capacity bytes and not alias message.
unsafe fn error_text(error: *mut c_char, capacity: u32, message: &str) {
    if !error.is_null() && capacity > 0 {
        let length = message.len().min(capacity as usize - 1);
        // SAFETY: The ABI caller provides this buffer; length is bounded by capacity.
        unsafe {
            ptr::copy_nonoverlapping(message.as_ptr(), error.cast::<u8>(), length);
            *error.add(length) = 0;
        }
    }
}

/// Translate recoverable failures and Rust panics without unwinding across C.
///
/// # Safety
/// The error pointer obeys error_text's caller-owned buffer contract.
unsafe fn boundary<F>(error: *mut c_char, capacity: u32, work: F) -> i32
where
    F: FnOnce() -> Result<(), String>,
{
    // SAFETY: The ABI error buffer contract is forwarded unchanged.
    unsafe { error_text(error, capacity, "") };
    let outcome = catch_unwind(AssertUnwindSafe(work));
    let message = match outcome {
        Ok(Ok(())) => return 0,
        Ok(Err(message)) => message,
        Err(_) => "native detector panicked".into(),
    };
    // SAFETY: The buffer remains valid throughout this call.
    unsafe { error_text(error, capacity, &message) };
    1
}

/// Report the unchanged public struct layout and mapped-detection ABI.
#[no_mangle]
pub extern "C" fn et_abi_version() -> u32 {
    2
}

/// Construct an owned native handle and fail explicitly on invalid settings.
///
/// # Safety
/// Config/output pointers and family bytes must be valid for their documented sizes.
#[no_mangle]
pub unsafe extern "C" fn et_create(
    config: *const Config,
    output: *mut *mut Handle,
    error: *mut c_char,
    capacity: u32,
) -> i32 {
    if !output.is_null() {
        // SAFETY: The caller provides a writable pointer-sized output slot.
        unsafe { *output = ptr::null_mut() };
    }
    // SAFETY: Error and configuration pointers obey the ABI contract.
    unsafe {
        boundary(error, capacity, || {
            if output.is_null() || config.is_null() {
                return Err("null configuration/output".into());
            }
            let config = &*config;
            if config.abi_version != 2
                || config.struct_size as usize != std::mem::size_of::<Config>()
            {
                return Err("configuration ABI mismatch".into());
            }
            if config.families.is_null()
                || !matches!(config.refine_edges, 0 | 1)
                || config.nthreads < 1
            {
                return Err("invalid detector configuration".into());
            }
            let settings = Settings {
                nthreads: config.nthreads as usize,
                refine_edges: config.refine_edges != 0,
                quad_decimate: config.quad_decimate,
                quad_sigma: config.quad_sigma,
                decode_sharpening: config.decode_sharpening,
            };
            settings.validate()?;
            let mut length = 0;
            while length <= 1024 && *config.families.add(length) != 0 {
                length += 1;
            }
            if length > 1024 {
                return Err("family list too long".into());
            }
            let names =
                std::str::from_utf8(slice::from_raw_parts(config.families.cast::<u8>(), length))
                    .map_err(|_| "family names must be UTF-8".to_string())?;
            let core = Detector::new(names, settings)?;
            *output = Box::into_raw(Box::new(Handle {
                busy: AtomicBool::new(false),
                core: UnsafeCell::new(core),
            }));
            Ok(())
        })
    }
}

/// Detect mapped pixels and lend result storage until the next detect or destroy.
///
/// # Safety
/// Handle must be live; image/map/output buffers must be valid. Destroy cannot race
/// this call. Input pixels must not be mutated while detection is running.
#[no_mangle]
pub unsafe extern "C" fn et_detect_mapped(
    handle: *mut Handle,
    image: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    source_map: *const f64,
    source_width: u32,
    source_height: u32,
    results: *mut *const Detection,
    count: *mut u32,
    error: *mut c_char,
    capacity: u32,
) -> i32 {
    if !results.is_null() {
        // SAFETY: The caller provides a writable pointer-sized output slot.
        unsafe { *results = ptr::null() };
    }
    if !count.is_null() {
        // SAFETY: The caller provides a writable count slot.
        unsafe { *count = 0 };
    }
    // SAFETY: The error pointer obeys the documented ABI buffer contract.
    unsafe {
        boundary(error, capacity, || {
            let handle = handle.as_ref().ok_or_else(|| "null detector".to_string())?;
            handle
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| "detector is already in use".to_string())?;
            let _guard = BusyGuard(&handle.busy);
            // UnsafeCell permits shared handle references, but the atomic guard
            // guarantees that only this call can borrow the mutable detector state.
            let core = &mut *handle.core.get();
            core.results.clear();
            if image.is_null() || results.is_null() || count.is_null() {
                return Err("invalid image dimensions, stride or output".into());
            }
            let span = image_span(width as usize, height as usize, stride as usize)?;
            let mapping = if source_map.is_null() {
                None
            } else {
                Some(*source_map.cast::<[f64; 9]>())
            };
            let shape = if mapping.is_some() || source_width != 0 || source_height != 0 {
                Some((source_height, source_width))
            } else {
                None
            };
            let pixels = slice::from_raw_parts(image, span);
            let image = mapped_image(
                pixels,
                width as usize,
                height as usize,
                stride as usize,
                mapping,
                shape,
            )?;
            let detected = core.detect(image);
            *count = detected.len() as u32;
            *results = if detected.is_empty() {
                ptr::null()
            } else {
                detected.as_ptr()
            };
            Ok(())
        })
    }
}

/// Detect an intentionally unmasked image through the same ABI 2 path.
///
/// # Safety
/// The handle, image and result pointers obey et_detect_mapped's contracts.
#[no_mangle]
pub unsafe extern "C" fn et_detect(
    handle: *mut Handle,
    image: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    results: *mut *const Detection,
    count: *mut u32,
    error: *mut c_char,
    capacity: u32,
) -> i32 {
    // SAFETY: Forward the exact caller-owned buffers with absent source geometry.
    unsafe {
        et_detect_mapped(
            handle,
            image,
            width,
            height,
            stride,
            ptr::null(),
            0,
            0,
            results,
            count,
            error,
            capacity,
        )
    }
}

/// Release a handle once, after all detection calls and result reads finish.
///
/// # Safety
/// A non-null handle must be owned and live; destroy must not race detection.
#[no_mangle]
pub unsafe extern "C" fn et_destroy(handle: *mut Handle) {
    if !handle.is_null() {
        // SAFETY: Ownership is transferred back exactly once by the ABI caller.
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(handle)) }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_does_not_cross_abi_and_releases_busy_guard() {
        assert_eq!(std::mem::size_of::<Config>(), 48);
        assert_eq!(std::mem::size_of::<Detection>(), 176);
        let busy = AtomicBool::new(false);
        let mut error = [0 as c_char; 7];
        // SAFETY: This local buffer has the declared capacity and no aliases.
        let status = unsafe {
            boundary(error.as_mut_ptr(), error.len() as u32, || {
                busy.store(true, Ordering::Release);
                let _guard = BusyGuard(&busy);
                panic!("injected boundary regression check");
            })
        };
        assert_eq!(status, 1);
        assert!(!busy.load(Ordering::Acquire));
        assert_eq!(error, [110, 97, 116, 105, 118, 101, 0]);
    }
}
