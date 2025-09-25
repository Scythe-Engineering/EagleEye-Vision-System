use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

#[derive(Clone)]
struct RoiTrack {
    center_x_px: f64,
    center_y_px: f64,
    size_px: f64,
    velocity_x_px: f64,
    velocity_y_px: f64,
    velocity_size_px: f64,
    missed_updates: u32,
}

impl RoiTrack {
    fn new(cx: f64, cy: f64, size: f64) -> Self {
        Self {
            center_x_px: cx,
            center_y_px: cy,
            size_px: size.max(1.0),
            velocity_x_px: 0.0,
            velocity_y_px: 0.0,
            velocity_size_px: 0.0,
            missed_updates: 0,
        }
    }
}

#[pyclass]
pub struct TemporalAcceleration {
    padding_factor: f64,
    max_missed_updates: u32,
    velocity_smoothing: f64,
    match_distance_px: f64,
    max_tracks: usize,
    tracks: Vec<RoiTrack>,
}

#[pymethods]
impl TemporalAcceleration {
    #[new]
    #[pyo3(signature = (padding_factor=0.3, max_missed_updates=2, velocity_smoothing=0.5, match_distance_px=80.0, max_tracks=16))]
    fn new(
        padding_factor: f64,
        max_missed_updates: u32,
        velocity_smoothing: f64,
        match_distance_px: f64,
        max_tracks: usize,
    ) -> Self {
        Self {
            padding_factor,
            max_missed_updates,
            velocity_smoothing,
            match_distance_px,
            max_tracks,
            tracks: Vec::new(),
        }
    }

    fn update_config(&mut self, json_config: &PyAny) -> PyResult<()> {
        if let Ok(pf) = json_config.get_item("padding_factor")?.extract::<f64>() {
            self.padding_factor = pf;
        }
        Ok(())
    }

    fn back_propagate_input(&mut self, detections: &PyAny) -> PyResult<()> {
        // Expect an iterable of (cx, cy, size)
        let mut parsed: Vec<(f64, f64, f64)> = Vec::new();
        if detections.is_none() {
            return Ok(());
        }

        if let Ok(seq) = detections.downcast::<PyList>() {
            for item in seq.iter() {
                if let Ok(t) = item.downcast::<PyTuple>() {
                    let len = t.len();
                    if len >= 3 {
                        let cx: f64 = t.get_item(0)?.extract()?;
                        let cy: f64 = t.get_item(1)?.extract()?;
                        let size: f64 = t.get_item(2)?.extract()?;
                        parsed.push((cx, cy, size));
                    }
                }
            }
        } else if let Ok(t) = detections.downcast::<PyTuple>() {
            if t.len() >= 3 {
                let cx: f64 = t.get_item(0)?.extract()?;
                let cy: f64 = t.get_item(1)?.extract()?;
                let size: f64 = t.get_item(2)?.extract()?;
                parsed.push((cx, cy, size));
            }
        }

        if parsed.is_empty() && self.tracks.is_empty() {
            return Ok(());
        }

        // Associate detections to tracks by nearest neighbor
        let mut assigned: Vec<Option<usize>> = vec![None; parsed.len()];
        let mut track_matched: Vec<bool> = vec![false; self.tracks.len()];

        for (det_idx, (cx, cy, _)) in parsed.iter().enumerate() {
            let mut best_idx: Option<usize> = None;
            let mut best_dist: f64 = f64::INFINITY;
            for (ti, tr) in self.tracks.iter().enumerate() {
                if track_matched[ti] {
                    continue;
                }
                let dx = cx - tr.center_x_px;
                let dy = cy - tr.center_y_px;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < best_dist && dist <= self.match_distance_px {
                    best_dist = dist;
                    best_idx = Some(ti);
                }
            }
            if let Some(ti) = best_idx {
                assigned[det_idx] = Some(ti);
                track_matched[ti] = true;
            }
        }

        // Update matched tracks and age unmatched
        let mut new_tracks: Vec<RoiTrack> = Vec::new();
        for (ti, mut tr) in self.tracks.clone().into_iter().enumerate() {
            if ti < track_matched.len() && track_matched[ti] {
                // find matching detection index
                let mut det_idx_opt: Option<usize> = None;
                for (di, a) in assigned.iter().enumerate() {
                    if let Some(at) = a {
                        if *at == ti { det_idx_opt = Some(di); break; }
                    }
                }
                if let Some(di) = det_idx_opt {
                    let (det_cx, det_cy, det_size) = parsed[di];
                    let new_vx = det_cx - tr.center_x_px;
                    let new_vy = det_cy - tr.center_y_px;
                    let new_vs = det_size - tr.size_px;
                    let alpha = self.velocity_smoothing.clamp(0.0, 1.0);
                    tr.velocity_x_px = (1.0 - alpha) * tr.velocity_x_px + alpha * new_vx;
                    tr.velocity_y_px = (1.0 - alpha) * tr.velocity_y_px + alpha * new_vy;
                    tr.velocity_size_px = (1.0 - alpha) * tr.velocity_size_px + alpha * new_vs;
                    tr.center_x_px = det_cx;
                    tr.center_y_px = det_cy;
                    tr.size_px = det_size.max(1.0);
                    tr.missed_updates = 0;
                    new_tracks.push(tr);
                } else {
                    tr.missed_updates = tr.missed_updates.saturating_add(1);
                    new_tracks.push(tr);
                }
            } else {
                tr.missed_updates = tr.missed_updates.saturating_add(1);
                if tr.missed_updates <= self.max_missed_updates { new_tracks.push(tr); }
            }
        }

        // Add unmatched detections as new tracks
        for (di, a) in assigned.iter().enumerate() {
            if a.is_none() && new_tracks.len() < self.max_tracks {
                let (cx, cy, size) = parsed[di];
                new_tracks.push(RoiTrack::new(cx, cy, size));
            }
        }

        self.tracks = new_tracks;
        Ok(())
    }

    fn process(&mut self, frame_width: i32, frame_height: i32) -> PyResult<Vec<(i32, i32, i32, i32)>> {
        if self.tracks.is_empty() {
            return Ok(vec![(0, 0, frame_width.max(0), frame_height.max(0))]);
        }

        // Predict forward
        for tr in self.tracks.iter_mut() {
            tr.center_x_px += tr.velocity_x_px;
            tr.center_y_px += tr.velocity_y_px;
            tr.size_px = (tr.size_px + tr.velocity_size_px).max(1.0);
        }

        let mut regions: Vec<(i32, i32, i32, i32)> = Vec::new();
        for tr in self.tracks.iter() {
            let half = 0.5 * tr.size_px * (1.0 + self.padding_factor);
            let left = (tr.center_x_px - half).floor() as i32;
            let top = (tr.center_y_px - half).floor() as i32;
            let right = (tr.center_x_px + half).ceil() as i32;
            let bottom = (tr.center_y_px + half).ceil() as i32;

            let l = left.max(0).min(frame_width);
            let t = top.max(0).min(frame_height);
            let r = right.max(0).min(frame_width);
            let b = bottom.max(0).min(frame_height);
            if r > l && b > t {
                regions.push((l, t, r, b));
            }
        }

        if regions.is_empty() {
            Ok(vec![(0, 0, frame_width.max(0), frame_height.max(0))])
        } else {
            Ok(regions)
        }
    }
}

#[pymodule]
fn temporal_acceleration(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TemporalAcceleration>()?;
    Ok(())
}
