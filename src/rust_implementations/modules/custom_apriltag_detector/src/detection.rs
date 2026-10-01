//! Owned detection fields shared by Python and the existing ABI 2 boundary.

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Detection {
    pub(crate) family_index: u32,
    pub(crate) tag_id: i32,
    pub(crate) hamming: i32,
    pub(crate) rotation: i32,
    pub(crate) decision_margin: f64,
    pub(crate) corners: [f64; 8],
    pub(crate) center: [f64; 2],
    pub(crate) homography: [f64; 9],
}
