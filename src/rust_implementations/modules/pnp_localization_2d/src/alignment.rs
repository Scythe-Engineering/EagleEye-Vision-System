//! Strict timestamp envelopes and selected-only circular NWU yaw validation.
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList},
};
use std::collections::BTreeMap;

pub struct Alignment {
    pub yaw: f64,
    pub mode: &'static str,
    pub delta: i64,
    pub gap: bool,
}
/// IEEE remainder is essential for parity at +/-pi and finite giant angles.
fn wrap(yaw: f64) -> f64 {
    libm::remainder(yaw, std::f64::consts::TAU)
}
/// Select exact, bracket, or outside-history nearest, without extrapolation.
fn select<F: Fn(i64) -> Result<f64, &'static str>>(
    times: &[i64],
    capture: i64,
    max_gap: i64,
    nearest: i64,
    heading: F,
) -> Result<Alignment, &'static str> {
    if capture <= 1 {
        return Err("missing_capture");
    }
    if times.is_empty() {
        return Err("missing_gyro");
    }
    let index = times.partition_point(|time| *time < capture);
    if index < times.len() && times[index] == capture {
        return Ok(Alignment {
            yaw: wrap(heading(capture)?),
            mode: "exact",
            delta: 0,
            gap: false,
        });
    }
    if index > 0 && index < times.len() {
        let before = times[index - 1];
        let after = times[index];
        let first = wrap(heading(before)?);
        let last = wrap(heading(after)?);
        if after - before > max_gap {
            return Err("gyro_gap_too_large");
        }
        let yaw =
            wrap(first + wrap(last - first) * (capture - before) as f64 / (after - before) as f64);
        return Ok(Alignment {
            yaw,
            mode: "interpolated",
            delta: after - before,
            gap: true,
        });
    }
    let time = if index == 0 {
        times[0]
    } else {
        times[times.len() - 1]
    };
    let delta = time - capture;
    if delta.abs() > nearest {
        return Err("stale_gyro");
    }
    Ok(Alignment {
        yaw: wrap(heading(time)?),
        mode: "nearest",
        delta,
        gap: false,
    })
}
/// Marshal only selected numeric yaw values; arbitrary unselected values are valid.
pub fn parse(
    samples: &Bound<'_, PyAny>,
    capture: i64,
    max_gap: i64,
    nearest: i64,
) -> Result<Alignment, &'static str> {
    if capture <= 1 {
        return Err("missing_capture");
    }
    let list = samples.cast::<PyList>().map_err(|_| "missing_gyro")?;
    if list.is_empty() {
        return Err("missing_gyro");
    }
    let mut values = BTreeMap::new();
    for sample in list.iter() {
        let dict = sample.cast::<PyDict>().map_err(|_| "invalid_gyro")?;
        if dict.len() != 2 {
            return Err("invalid_gyro");
        }
        let timestamp = dict
            .get_item("timestamp_us")
            .map_err(|_| "invalid_gyro")?
            .ok_or("invalid_gyro")?;
        if !timestamp.is_exact_instance_of::<PyInt>() {
            return Err("invalid_gyro");
        }
        let time = timestamp.extract::<i64>().map_err(|_| "invalid_gyro")?;
        if time <= 1 {
            return Err("invalid_gyro");
        }
        let value = dict
            .get_item("value")
            .map_err(|_| "invalid_gyro")?
            .ok_or("invalid_gyro")?;
        values.insert(time, value);
    }
    let times: Vec<i64> = values.keys().copied().collect();
    select(&times, capture, max_gap, nearest, |time| {
        let value = &values[&time];
        if value.is_instance_of::<PyBool>()
            || !(value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>())
        {
            return Err("invalid_gyro");
        }
        let yaw = value.extract::<f64>().map_err(|_| "invalid_gyro")?;
        if !yaw.is_finite() {
            return Err("invalid_gyro");
        }
        Ok(yaw)
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn circular_edges_and_selection() {
        let times = [10, 20];
        let result = select(&times, 15, 10, 2, |time| {
            Ok(if time == 10 { 3.1 } else { -3.1 })
        })
        .unwrap();
        assert!((result.yaw.abs() - std::f64::consts::PI).abs() < 1e-15);
        assert_eq!(
            select(&times, 15, 9, 20, |_| Ok(0.)).err(),
            Some("gyro_gap_too_large")
        );
        assert_eq!(
            select(&times, 30, 100, 2, |_| panic!("stale must not inspect yaw")).err(),
            Some("stale_gyro")
        );
        assert_eq!(
            select(&times, 10, 10, 2, |t| if t == 10 {
                Ok(1e300)
            } else {
                Err("invalid_gyro")
            })
            .unwrap()
            .yaw,
            libm::remainder(1e300, std::f64::consts::TAU)
        );
        assert_eq!(wrap(-std::f64::consts::PI), -std::f64::consts::PI);
        assert!(wrap(-0.).is_sign_negative());
    }
}
