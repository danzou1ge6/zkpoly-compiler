use super::*;
use std::{fmt::Debug, io::Write};
use zkpoly_common::{arith::ExprId, digraph::internal::Digraph};

pub fn write_graph<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    output_vid: VertexId,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(
        g,
        writer,
        None,
        seq,
        false,
        |_, _| None,
        |_, _| None,
        |_, _| None,
    )
}

pub fn write_optinally_typed_graph<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<Ty>>>,
    error_vid: VertexId,
    output_vid: VertexId,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(
        g,
        writer,
        None,
        seq,
        false,
        |vid, vertex| {
            if vid == error_vid {
                Some("#e74c3c") // red
            } else {
                if vertex.typ().is_some() {
                    Some("#2ecc71") // green
                } else {
                    None
                }
            }
        },
        |_, _| None,
        |_, _| None,
    )
}

pub fn write_graph_with_vertices_colored<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    output_vid: VertexId,
    writer: &mut impl Write,
    override_color: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<&'static str>,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(
        g,
        writer,
        None,
        seq,
        false,
        override_color,
        |_, _| None,
        |_, _| None,
    )
}

pub fn write_graph_with_seq<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    writer: &mut impl Write,
    seq: impl Iterator<Item = VertexId> + Clone,
) -> std::io::Result<()> {
    write_graph_with_optional_seq(
        g,
        writer,
        None,
        seq,
        true,
        |_, _| None,
        |_, _| None,
        |_, _| None,
    )
}

pub(crate) fn format_source_info<'s>(src: &SourceInfo<'s>) -> String {
    let mut src_string = String::new();
    if let Some(name) = &src.name {
        src_string.push_str(&format!("{} ", name));
    }
    if let Some(loc) = src.location.first() {
        src_string.push_str(&format!("{}:{}:{}", loc.file, loc.line, loc.column));
    }
    src_string
}

/// Write the computation graph in DOT format
pub fn write_graph_with_optional_seq<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    writer: &mut impl Write,
    _first_shown: Option<VertexId>,
    seq: impl Iterator<Item = VertexId> + Clone,
    print_seq: bool,
    override_color: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<&'static str>,
    extra_tooltip: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<String>,
    edge_tooltip: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<Vec<(VertexId, String)>>,
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

            let mut label = format!("{}: ArithGraph", vid.0,);

            if print_seq {
                label.push_str(&format!("({})", i));
            }

            write!(writer, "    label = \"{}\"", label)?;
            write!(
                writer,
                "    tooltip = \"{:?} {} @{}\"",
                vertex.typ(),
                extra_tooltip(vid, vertex).unwrap_or_default(),
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
            use super::DevicePreference::*;
            let style = match vertex.node().device() {
                Gpu => "filled",
                Cpu => "solid",
                PreferGpu => "dashed",
            };
            (get_node_color(vertex.node()), style)
        };
        // Write node with attributes
        writeln!(
            writer,
            "  {} [id = \"v{}\",label=\"{}:{}\", tooltip=\"{:?} {}\\n@{}\", style=\"{}\", fillcolor=\"{}\"]",
            vid.0,
            vid.0,
            vid.0,
            label,
            vertex.typ(),
            extra_tooltip(vid, vertex).unwrap_or_default(),
            format_source_info(vertex.src()),
            style,
            color
        )?;
    }

    // Write edges
    for to_vid in seq {
        let vertex = g.vertex(to_vid);

        let tooltips = edge_tooltip(to_vid, vertex);

        if let template::VertexNode::Arith { arith, .. } = vertex.node() {
            arith::pretty_print::print_subgraph_edges(
                arith,
                &format!("_{}_arith_", to_vid.0),
                format!("{}", to_vid.0),
                |ivid| format!("{}", ivid.0),
                writer,
                tooltips,
            )?;
            continue;
        }

        for (from_vid, label) in format_labeled_uses(vertex.node()).into_iter() {
            let tooltip = tooltips.as_ref().map_or_else(
                || "",
                |xs| {
                    xs.iter()
                        .find(|(vid, _)| *vid == from_vid)
                        .map(|(_, tooltip)| tooltip)
                        .unwrap()
                },
            );
            writeln!(
                writer,
                "  {} -> {} [class = \"v{}-neighbour v{}-neighbour\"headlabel=\"{}\", labeldistance=2, edgetooltip=\"{}\"]",
                from_vid.0, to_vid.0,
                from_vid.0, to_vid.0,
                label,
                tooltip
            )?;
        }
    }

    writeln!(writer, "}}")
}

pub(crate) fn format_node_label<'s, Vid: UsizeId + Debug>(
    vertex_node: &template::VertexNode<
        Vid,
        arith::ArithGraph<Vid, ExprId>,
        ConstantId,
        user_function::Id,
    >,
) -> String {
    use template::VertexNode::*;
    match vertex_node {
        NewPoly(deg, ..) => format!("NewPoly(deg={})", deg),
        Constant(cid) => format!("Constant({:?})", cid),
        Extend(_, deg) => format!("Extend(deg={})", deg),
        SingleArith(arith::Arith::Bin(op, ..)) => {
            format!("SingleArith({:?})", op)
        }
        SingleArith(arith::Arith::Unr(op, ..)) => {
            format!("SingleArith({:?})", op)
        }
        Arith { chunking, .. } => {
            if chunking.is_some() {
                format!("Arith(chunked)")
            } else {
                String::from("Arith")
            }
        }
        PolyPermute(_input, _table, usable_len) => {
            format!("PolyPermute({})", usable_len)
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
        AssertEq(..) => "AssertEq".to_string(),
        Print(_, label) => format!("Print({})", label),
    }
}

pub(crate) fn format_labeled_uses<'s, Vid: UsizeId + Debug>(
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
        AssertEq(a, b, _msg) => {
            vec![(*a, "a".to_string()), (*b, "".to_string())]
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
