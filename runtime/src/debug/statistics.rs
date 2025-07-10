use std::{
    collections::{BTreeMap, BTreeSet},
    ops::AddAssign,
};

use super::Log;
use crate::{
    functions::FuncMeta,
    instructions::{Instruction, InstructionNode},
};
use plotters::{
    prelude::{self as plt, IntoSegmentedCoord},
    style::Color,
};
use zkpoly_common::devices::DeviceType;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Arith,
    MemoryManagement,
    Transfer { from: DeviceType, to: DeviceType },
    F(String),
    Record,
    Fork,
    Other,
}

impl std::fmt::Debug for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Category::*;
        match self {
            Arith => write!(f, "Arith"),
            MemoryManagement => write!(f, "MemoryManagement"),
            Transfer { from, to } => {
                let device_str = |d: &DeviceType| -> String {
                    use DeviceType::*;
                    match d {
                        CPU => "CPU".to_string(),
                        GPU { device_id } => format!("GPU({device_id})"),
                        Disk => "Disk".to_string(),
                    }
                };
                write!(f, "{}->{}", device_str(from), device_str(to))
            }
            F(s) => write!(f, "{s}"),
            Record => write!(f, "Record"),
            Fork => write!(f, "Fork"),
            Other => write!(f, "Other"),
        }
    }
}

impl Category {
    pub fn of(inst: &Instruction, function_meta: &Option<FuncMeta>) -> Self {
        use InstructionNode::*;
        match inst.node() {
            FuncCall { .. } => {
                let fname = &function_meta.as_ref().unwrap().name;
                if fname.starts_with("fused_arith") {
                    Self::Arith
                } else {
                    Self::F(fname.clone())
                }
            }
            Allocate { .. } => Self::MemoryManagement,
            Deallocate { .. } => Self::MemoryManagement,
            Transfer {
                src_device,
                dst_device,
                ..
            } => Self::Transfer {
                from: src_device.clone(),
                to: dst_device.clone(),
            },
            Fork { .. } => Self::Fork,
            Record { .. } => Self::Record,
            _ => Self::Other,
        }
    }
}

pub fn plot_percentage<DB: plt::DrawingBackend>(
    log: &Log,
    cb: &mut plt::ChartBuilder<'_, '_, DB>,
    top_k: usize,
) -> Result<(), plt::DrawingAreaErrorKind<DB::ErrorType>> {
    let insts = log
        .executed_instructions
        .iter()
        .map(|ie| {
            (
                Category::of(&ie.instruction, &ie.function_meta),
                ie.duration_nanos() as f64 / 1000_000.0,
            )
        })
        .collect::<Vec<_>>();
    let categories = insts.iter().fold(BTreeMap::new(), |mut acc, (c, v)| {
        acc.entry(c.clone()).or_insert(0.0).add_assign(v);
        acc
    });

    let mut categories = categories.into_iter().collect::<Vec<_>>();
    categories.sort_by(|(_, lhs), (_, rhs)| f64::total_cmp(lhs, rhs).reverse());

    let x_range = 0.0..(categories
        .iter()
        .fold(f64::NAN, |max, (_, val)| val.max(max))
        * 1.1);

    let c = categories
        .iter()
        .map(|(k, _)| k.clone())
        .collect::<Vec<_>>();
    let mut cc = cb.build_cartesian_2d(x_range, c.into_segmented())?;

    cc.configure_mesh().light_line_style(&plt::WHITE).draw()?;
    cc.draw_series(
        plt::Histogram::horizontal(&cc)
            .style(plt::BLUE.filled())
            .margin(20)
            .data(categories.iter().map(|(k, v)| (k, *v))),
    )?;

    Ok(())
}
