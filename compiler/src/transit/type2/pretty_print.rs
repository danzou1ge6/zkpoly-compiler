use crate::transit::Locations;

use super::*;
use std::{fmt::Debug, io::Write};
use zkpoly_common::{arith::ExprId, digraph::internal::Digraph};

pub fn write_graph<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    output_vid: VertexId,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(g, writer, seq, false, |_, _| None)
}

pub fn write_optinally_typed_graph<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<Ty>>>,
    error_vid: VertexId,
    output_vid: VertexId,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(g, writer, seq, false, |vid, vertex| {
        if vid == error_vid {
            Some("#e74c3c") // red
        } else {
            if vertex.typ().is_some() {
                Some("#2ecc71") // green
            } else {
                None
            }
        }
    })
}

pub fn write_graph_with_vertices_colored<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    output_vid: VertexId,
    writer: &mut impl Write,
    override_color: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<&'static str>,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(g, writer, seq, false, override_color)
}

pub fn write_graph_with_seq<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    writer: &mut impl Write,
    seq: impl Iterator<Item = VertexId> + Clone,
) -> std::io::Result<()> {
    write_graph_with_optional_seq(g, writer, seq, true, |_, _| None)
}

fn format_source_info<'s>(src: &SourceInfo<'s>) -> String {
    let loc = match &src.location {
        Locations::Multi(locs) => &locs[0],
        Locations::Single(loc) => loc,
    };
    format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
}

/// Write the computation graph in DOT format
fn write_graph_with_optional_seq<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    writer: &mut impl Write,
    seq: impl Iterator<Item = VertexId> + Clone,
    print_seq: bool,
    override_color: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<&'static str>,
) -> std::io::Result<()> {
    // Write the DOT header
    writeln!(writer, "digraph ComputationGraph {{")?;
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
    for (i, vid) in seq.clone().enumerate() {
        let vertex = g.vertex(vid);

        if let template::VertexNode::Arith { arith, .. } = vertex.node() {
            writeln!(writer, "  subgraph cluster_arith{} {{", vid.0)?;
            writeln!(
                writer,
                "    label = \"{}\"",
                format_source_info(vertex.src())
            )?;
            let color = override_color(vid, vertex).unwrap_or("blue");
            writeln!(writer, "    color = \"{}\"", color)?;
            writeln!(writer, "    style = dashed")?;
            arith::pretty_print::print_subgraph_vertices(
                arith,
                &format!("_{}_arith_", vid.0),
                format!("{}", vid.0),
                writer,
            )?;
            writeln!(writer, "  }}")?;
            continue;
        }

        let mut label = format_node_label(vertex.node());

        if print_seq {
            label.push_str(&format!("({})", i));
        }

        // Get node style and color
        let (color, style) = if let Some(color) = override_color(vid, vertex) {
            (color, "filled")
        } else {
            let style = match vertex.node().device() {
                Device::Gpu => "filled",
                Device::Cpu => "solid",
                Device::PreferGpu => "dashed",
            };
            (get_node_color(vertex.node()), style)
        };
        // Write node with attributes
        writeln!(
            writer,
            "  {} [id = \"v{}\",label=\"{}:{}\", tooltip=\"{:?}\\n@{}\", style=\"{}\", fillcolor=\"{}\"]",
            vid.0,
            vid.0,
            vid.0,
            label,
            vertex.typ(),
            format_source_info(vertex.src()),
            style,
            color
        )?;
    }

    // Write edges
    for to_vid in seq {
        let vertex = g.vertex(to_vid);

        if let template::VertexNode::Arith { arith, .. } = vertex.node() {
            arith::pretty_print::print_subgraph_edges(
                arith,
                &format!("_{}_arith_", to_vid.0),
                format!("{}", to_vid.0),
                |ivid| format!("{}", ivid.0),
                writer,
            )?;
            continue;
        }

        for (from_vid, label) in format_labeled_uses(vertex.node()) {
            writeln!(
                writer,
                "  {} -> {} [class = \"v{}-neighbour v{}-neighbour\"headlabel=\"{}\", labeldistance=2]",
                from_vid.0, to_vid.0,
                from_vid.0, to_vid.0,
                label
            )?;
        }
    }

    writeln!(writer, "}}")
}

pub fn format_node_label<'s, Vid: UsizeId + Debug>(
    vertex_node: &template::VertexNode<
        Vid,
        arith::ArithGraph<Vid, ExprId>,
        ConstantId,
        user_function::Id,
    >,
) -> String {
    use template::VertexNode::*;
    match vertex_node {
        NewPoly(deg, ..) => format!("NewPoly\\n(deg={})", deg),
        Constant(_) => String::from("Constant"),
        Extend(_, deg) => format!("Extend\\n(deg={})", deg),
        SingleArith(arith::Arith::Bin(op, ..)) => {
            format!("SingleArith({:?})", op)
        }
        SingleArith(arith::Arith::Unr(op, ..)) => {
            format!("SingleArith({:?})", op)
        }
        Arith { chunking, .. } => {
            if chunking.is_some() {
                format!("Arith\\n(chunked)")
            } else {
                String::from("Arith")
            }
        }
        Entry(id) => format!("Entry({:?})", id),
        Return(_) => String::from("Return"),
        Ntt { from, to, .. } => format!("NTT({:?} -> {:?})", from, to),
        RotateIdx(_, idx) => format!("Rotate({})", idx),
        Slice(_, start, end) => format!("Slice[{}:{}]", start, end),
        Interpolate { .. } => String::from("Interpolate"),
        Blind(_, left, right) => format!("Blind[{}:{}]", left, right),
        Array(_) => String::from("Array"),
        AssmblePoly(deg, _) => format!("AssemblePoly(deg={})", deg),
        Msm { .. } => String::from("MSM"),
        HashTranscript { typ, .. } => format!("Hash({:?})", typ),
        SqueezeScalar(_) => String::from("SqueezeScalar"),
        TupleGet(_, idx) => format!("TupleGet({})", idx),
        ArrayGet(_, idx) => format!("ArrayGet({})", idx),
        UserFunction(id, _) => format!("UserFunc({:?})", id),
        KateDivision(..) => String::from("KateDivision"),
        EvaluatePoly { .. } => String::from("EvalPoly"),
        BatchedInvert(_) => String::from("BatchInvert"),
        ScanMul { .. } => String::from("ScanMul"),
        DistributePowers { .. } => String::from("DistPowers"),
        ScalarInvert { .. } => String::from("ScalarInvert"),
        IndexPoly(_, idx) => format!("IndexPoly({})", idx),
    }
}

fn format_labeled_uses<'s, Vid: UsizeId + Debug>(
    vertex_node: &template::VertexNode<
        Vid,
        arith::ArithGraph<Vid, ExprId>,
        ConstantId,
        user_function::Id,
    >,
) -> Vec<(Vid, String)> {
    use template::VertexNode::*;
    match vertex_node {
        Interpolate { xs, ys } => xs
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| (x, format!("x{}", i)))
            .chain(
                ys.iter()
                    .copied()
                    .enumerate()
                    .map(|(i, y)| (y, format!("y{}", i))),
            )
            .collect(),
        Array(xs) | AssmblePoly(_, xs) => xs
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| (x, format!("{}", i)))
            .collect(),
        Msm { polys, points, .. } => polys
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| (x, format!("poly{}", i)))
            .chain(
                points
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, x)| (x, format!("point{}", i))),
            )
            .collect(),
        HashTranscript {
            transcript, value, ..
        } => vec![
            (*transcript, "transcript".to_string()),
            (*value, "value".to_string()),
        ],
        KateDivision(lhs, rhs) => vec![(*lhs, "lhs".to_string()), (*rhs, "rhs".to_string())],
        EvaluatePoly { poly, at } => vec![(*poly, "poly".to_string()), (*at, "at".to_string())],
        ScanMul { x0, poly } => vec![(*x0, "x0".to_string()), (*poly, "poly".to_string())],
        DistributePowers { poly, powers } => {
            vec![(*poly, "poly".to_string()), (*powers, "powers".to_string())]
        }
        _ => vertex_node.uses().map(|u| (u, String::new())).collect(),
    }
}

pub fn get_node_color<I, A, C, E>(node: &template::VertexNode<I, A, C, E>) -> &'static str {
    use template::VertexNode::*;
    match node {
        NewPoly(..) => "#A5D6A7",  // Light green
        Arith { .. } => "#90CAF9", // Light blue
        Ntt { .. } => "#FFE082",   // Light yellow
        Msm { .. } => "#F48FB1",   // Light pink
        Return(..) => "#B39DDB",   // Light purple
        Entry(..) => "#80CBC4",    // Light teal
        _ => "#FFFFFF",            // White for other nodes
    }
}

#[cfg(test)]
mod tests {
    use std::panic::Location;

    use crate::transit::Locations;

    use super::*;
    use zkpoly_common::{arith::ArithGraph, typ::PolyType};
    use zkpoly_runtime::transcript::Challenge255;

    use halo2curves::bn256;
    use zkpoly_runtime::args::RuntimeType;
    use zkpoly_runtime::transcript::Blake2bWrite;

    #[derive(Debug, Clone)]
    pub struct MyRuntimeType;

    impl RuntimeType for MyRuntimeType {
        type Field = bn256::Fr;
        type PointAffine = bn256::G1Affine;
        type Challenge = Challenge255<bn256::G1Affine>;
        type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
    }

    #[test]
    fn test_pretty_print() -> std::io::Result<()> {
        // Create a simple test program
        use digraph::internal::Digraph;
        let mut g: Digraph<VertexId, Vertex<'_, MyRuntimeType>> = Digraph::new();

        // Create vertices
        let v1 = g.add_vertex(Vertex::new(
            VertexNode::NewPoly(64, PolyInit::Ones, PolyType::Coef),
            Typ::Poly((PolyType::Coef, 64)),
            SourceInfo::new(Locations::Single(*Location::caller()), None),
        ));

        let v2 = g.add_vertex(Vertex::new(
            VertexNode::Ntt {
                s: v1,
                from: PolyType::Coef,
                to: PolyType::Lagrange,
                alg: NttAlgorithm::Undecieded,
            },
            Typ::Poly((PolyType::Lagrange, 64)),
            SourceInfo::new(Locations::Single(*Location::caller()), None),
        ));

        let cg: transit::Cg<
            VertexId,
            transit::Vertex<
                template::VertexNode<
                    VertexId,
                    ArithGraph<VertexId, arith::ExprId>,
                    ConstantId,
                    ast::lowering::UserFunctionId,
                >,
                typ::template::Typ<_, (PolyType, u64)>,
                SourceInfo<'_>,
            >,
        > = Cg { g, output: v2 };

        let mut buffer = Vec::new();
        write_graph(&cg.g, v2, &mut buffer)?;

        // Convert to string and verify basic content
        let output = String::from_utf8(buffer).unwrap();

        print!("{}", output);
        // assert!(output.contains("digraph ComputationGraph"));
        // assert!(output.contains("NewPoly"));
        // assert!(output.contains("NTT"));
        // assert!(output.contains("Arith"));
        // assert!(output.contains("Return"));

        Ok(())
    }
}
