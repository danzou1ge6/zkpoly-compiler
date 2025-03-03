use super::*;
use crate::transit::type2;
use std::io::Write;
use zkpoly_runtime::args::RuntimeType;

/// Write the instruction sequence in DOT format
pub fn write_graph<'s, Rt: RuntimeType>(
    chunk: &Chunk<'s, Rt>,
    mut writer: impl Write,
) -> std::io::Result<()> {
    // Write the DOT header
    writeln!(writer, "digraph InstructionGraph {{")?;
    writeln!(writer, "  // Graph settings")?;
    writeln!(writer, "  graph [")?;
    writeln!(writer, "    fontname = \"Helvetica\"")?;
    writeln!(writer, "    fontsize = 11")?;
    writeln!(writer, "  ]")?;

    // Node settings
    writeln!(writer, "  // Node settings")?;
    writeln!(writer, "  node [")?;
    writeln!(writer, "    fontname = \"Helvetica\"")?;
    writeln!(writer, "    fontsize = 11")?;
    writeln!(writer, "    shape = \"box\"")?;
    writeln!(writer, "    style = \"rounded\"")?;
    writeln!(writer, "  ]")?;

    // Edge settings
    writeln!(writer, "  // Edge settings")?;
    writeln!(writer, "  edge [")?;
    writeln!(writer, "    fontname = \"Helvetica\"")?;
    writeln!(writer, "    fontsize = 11")?;
    writeln!(writer, "  ]")?;

    // Write nodes
    for (idx, instruction) in chunk.iter_instructions() {
        let label = format_node_label::<Rt>(instruction);
        let track =
            instruction.track(|id| *chunk.register_devices.get(&id).unwrap_or(&Device::Cpu));
        let color = get_node_color(&instruction.node);
        let style = get_track_style(track);

        // Write node with attributes
        writeln!(
            writer,
            "  {} [label=\"{}, instruction id: {}\", style=\"{}\", fillcolor=\"{}\"]",
            idx.0, label, idx.0, style, color
        )?;
    }

    // Write edges for data dependencies
    for (from_idx, instruction) in chunk.iter_instructions() {
        for use_id in instruction.uses() {
            writeln!(writer, "  {} -> {}", use_id.0, from_idx.0,)?;
        }
    }

    writeln!(writer, "}}")
}

fn format_node_label<'s, Rt: RuntimeType>(instruction: &Instruction<'s>) -> String {
    use template::InstructionNode::*;
    match &instruction.node {
        Type2 { vertex, .. } => type2::pretty_print::format_node_label(vertex),

        GpuMalloc { id, addr } => format!("GPU Malloc\nR{} @ A{}", id.0, addr.0),
        GpuFree { id } => format!("GPU Free\nR{}", id.0),
        CpuMalloc { id, size } => format!("CPU Malloc\nR{} ({:?})", id.0, size),
        CpuFree { id } => format!("CPU Free\nR{}", id.0),
        StackFree { id } => format!("Stack Free\nR{}", id.0),
        Tuple { id, oprands } => {
            let oprs = oprands
                .iter()
                .map(|x| format!("R{}", x.0))
                .collect::<Vec<_>>()
                .join(", ");
            format!("Tuple\nR{} = ({})", id.0, oprs)
        }
        Transfer { id, from } => format!("Transfer\nR{} <- R{}", id.0, from.0),
        Move { id, from } => format!("Move\nR{} <- R{}", id.0, from.0),
        SetPolyMeta {
            id,
            from,
            offset,
            len,
        } => {
            format!(
                "SetPolyMeta\nR{} <- R{}\noffset={},len={}",
                id.0, from.0, offset, len
            )
        }
    }
}

fn get_node_color(node: &InstructionNode) -> &'static str {
    match node {
        // 对于Type2节点，直接调用type2的get_node_color
        InstructionNode::Type2 { vertex, .. } => type2::pretty_print::get_node_color(vertex),
        InstructionNode::GpuMalloc { .. } => "#FFF59D", // Light yellow
        InstructionNode::GpuFree { .. } => "#FFF59D",   // Light yellow
        InstructionNode::CpuMalloc { .. } => "#FFE0B2", // Light orange
        InstructionNode::CpuFree { .. } => "#FFE0B2",   // Light orange
        InstructionNode::StackFree { .. } => "#FFE0B2", // Light orange
        InstructionNode::Tuple { .. } => "#A5D6A7",     // Light green
        InstructionNode::Transfer { .. } => "#F48FB1",  // Light pink
        InstructionNode::Move { .. } => "#90CAF9",      // Light blue
        InstructionNode::SetPolyMeta { .. } => "#B39DDB", // Light purple
    }
}

fn get_track_style(track: Track) -> &'static str {
    match track {
        Track::MemoryManagement => "filled",
        Track::CoProcess => "filled",
        Track::Gpu => "filled",
        Track::Cpu => "filled",
        Track::ToGpu => "filled,dashed",
        Track::FromGpu => "filled,dashed",
        Track::GpuMemory => "filled",
    }
}
