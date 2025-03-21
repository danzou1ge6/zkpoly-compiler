use std::io::Write;

use super::{Arith, ArithGraph, ExprId, Operation};

pub fn print_subgraph_vertices<I: Copy>(
    ag: &ArithGraph<I, ExprId>,
    vertex_name_prefix: &str,
    all_output_id: String,
    writer: &mut impl Write,
) -> std::io::Result<()> {
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
            vertex_name_prefix, v.0, vertex_name_prefix, v.0, label
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

pub fn print_subgraph_edges<I: Copy + Eq>(
    ag: &ArithGraph<I, ExprId>,
    vertex_name_prefix: &str,
    all_output_id: String,
    vid: impl Fn(I) -> String,
    writer: &mut impl Write,
    edge_tooltips: Option<Vec<(I, String)>>,
) -> std::io::Result<()> {
    for v in ag.g.vertices() {
        let op = &ag.g.vertex(v).op;
        // Write internal edges
        for (us, label) in labeled_uses(op).into_iter() {
            writeln!(
                writer,
                "  {}{} -> {}{} [class = \"v{}{}-neighbour v{}{}-neighbour\", headlabel=\"{}\", labeldistance=2]",
                vertex_name_prefix, us.0, vertex_name_prefix, v.0,
                vertex_name_prefix, us.0, vertex_name_prefix, v.0,
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
                v.0,
                vid(*outer_id),
                vertex_name_prefix,
                v.0,
                tooltip
            )?;
        }
    }

    for (i, &output) in ag.outputs.iter().enumerate() {
        writeln!(
            writer,
            "  {}{} -> {} [class = \"v{}{}-neighbour v{}-neighbour\", headlabel=\"{}\", labeldistance=2]",
            vertex_name_prefix, output.0, all_output_id,
            vertex_name_prefix, output.0, all_output_id, i
        )?;
    }

    Ok(())
}

fn format_node_label<I>(op: &Operation<I, ExprId>) -> String {
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

fn labeled_uses<I>(op: &Operation<I, ExprId>) -> Vec<(ExprId, &'static str)> {
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
