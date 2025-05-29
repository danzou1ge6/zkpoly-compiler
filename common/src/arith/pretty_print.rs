use std::io::Write;

use crate::heap::UsizeId;

use super::{Arith, ArithGraph, Operation};

pub fn print_subgraph_vertices<I: Copy, Ii>(
    ag: &ArithGraph<I, Ii>,
    vertex_name_prefix: &str,
    all_output_id: String,
    writer: &mut impl Write,
) -> std::io::Result<()>
where
    Ii: UsizeId,
{
    // Node settings
    writeln!(writer, "    // Node settings")?;
    writeln!(writer, "    node [")?;
    writeln!(writer, "      shape = \"box\"")?;
    writeln!(writer, "      style = \"rounded\"")?;
    writeln!(writer, "    ]")?;

    // Write nodes
    for v in ag.g.vertices() {
        let op = &ag.g.vertex(v).op;
        let label = format_node_label(op);

        writeln!(
            writer,
            "    {}{} [id = \"v{}{}\", label=\"{}\", style=solid]",
            vertex_name_prefix,
            v.into(),
            vertex_name_prefix,
            v.into(),
            label
        )?;
    }

    // Write output collection node
    writeln!(
        writer,
        "    {} [id = \"v{}\", label=\"{}\", style=solid]",
        all_output_id, all_output_id, "AllOutputs"
    )?;

    Ok(())
}

pub fn print_subgraph_edges<I: Copy + Eq, Ii>(
    ag: &ArithGraph<I, Ii>,
    vertex_name_prefix: &str,
    all_output_id: String,
    vid: impl Fn(I) -> String,
    writer: &mut impl Write,
    edge_tooltips: Option<Vec<(I, String)>>,
) -> std::io::Result<()>
where
    Ii: UsizeId,
{
    for v in ag.g.vertices() {
        let op = &ag.g.vertex(v).op;
        // Write internal edges
        for (us, label) in labeled_uses(op).into_iter() {
            writeln!(
                writer,
                "  {}{} -> {}{} [class = \"v{}{}-neighbour v{}{}-neighbour\", headlabel=\"{}\", labeldistance=2]",
                vertex_name_prefix, us.into(), vertex_name_prefix, v.into(),
                vertex_name_prefix, us.into(), vertex_name_prefix, v.into(),
                label
            )?;
        }
        // Write external in-edges
        if let Operation::Input { outer_id, .. } = op {
            let tooltip = edge_tooltips.as_ref().map_or_else(
                || "",
                |xs| {
                    xs.iter()
                        .find(|(x, _)| x == outer_id)
                        .map(|(_, x)| x)
                        .unwrap()
                },
            );
            writeln!(
                writer,
                "  {} -> {}{} [class = \"v{}-neighbour v{}{}-neighbour\", edgetooltip = \"{}\"]",
                vid(*outer_id),
                vertex_name_prefix,
                v.into(),
                vid(*outer_id),
                vertex_name_prefix,
                v.into(),
                tooltip
            )?;
        }
    }

    for (i, &output) in ag.outputs.iter().enumerate() {
        writeln!(
            writer,
            "  {}{} -> {} [class = \"v{}{}-neighbour v{}-neighbour\", headlabel=\"{}\", labeldistance=2]",
            vertex_name_prefix, output.into(), all_output_id,
            vertex_name_prefix, output.into(), all_output_id, i
        )?;
    }

    Ok(())
}

pub(crate) fn format_node_label<I, Ii>(op: &Operation<I, Ii>) -> String {
    match op {
        Operation::Arith(Arith::Bin(op, ..)) => {
            format!("{:?}", op)
        }
        Operation::Arith(Arith::Unr(op, ..)) => {
            format!("{:?}", op)
        }
        Operation::Input {
            typ, mutability, ..
        } => {
            format!("Input({:?}, {:?})", typ, mutability)
        }
        Operation::Output { typ, .. } => {
            format!("Output({:?})", typ)
        }
        Operation::Todo => {
            format!("Todo")
        }
    }
}

pub(crate) fn labeled_uses<I, Ii: Copy>(op: &Operation<I, Ii>) -> Vec<(Ii, &'static str)> {
    match op {
        Operation::Arith(Arith::Bin(_, lhs, rhs)) => vec![(*lhs, "lhs"), (*rhs, "rhs")],
        Operation::Arith(Arith::Unr(_, expr)) => vec![(*expr, "")],
        Operation::Input { .. } => vec![],
        Operation::Output {
            store_node,
            in_node,
            ..
        } => in_node
            .iter()
            .map(|e| (*e, "in"))
            .chain(std::iter::once((*store_node, "store")))
            .collect(),
        Operation::Todo => vec![],
    }
}
