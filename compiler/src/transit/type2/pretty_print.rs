use crate::transit::Cg;

use super::*;
use std::{fmt::Debug, io::Write};
use zkpoly_common::arith::ExprId;
use zkpoly_runtime::args::RuntimeType;

pub fn write_graph<'s, Rt: RuntimeType>(
    cg: &Cg<VertexId, Vertex<'s, Rt>>,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = cg.g.vertices();
    write_graph_with_optional_seq(cg, writer, seq, false)
}

pub fn write_graph_with_seq<'s, Rt: RuntimeType>(
    cg: &Cg<VertexId, Vertex<'s, Rt>>,
    writer: &mut impl Write,
    seq: impl Iterator<Item = VertexId>,
) -> std::io::Result<()> {
    write_graph_with_optional_seq(cg, writer, seq, true)
}

/// Write the computation graph in DOT format
fn write_graph_with_optional_seq<'s, Rt: RuntimeType>(
    cg: &Cg<VertexId, Vertex<'s, Rt>>,
    writer: &mut impl Write,
    seq: impl Iterator<Item = VertexId>,
    print_seq: bool,
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
    for (i, vid) in seq.enumerate() {
        let vertex = cg.g.vertex(vid);
        let mut label = format_node_label::<VertexId, Rt>(vertex.node());

        if print_seq {
            label.push_str(&format!("({})", i));
        }

        // Get node style and color
        let style = match vertex.device() {
            Device::Gpu => "filled",
            Device::Cpu => "solid",
            Device::PreferGpu => "dashed",
        };
        let color = get_node_color(vertex.node());

        // Write node with attributes
        writeln!(
            writer,
            "  {} [label=\"{}\", style=\"{}\", fillcolor=\"{}\"]",
            vid.0, label, style, color
        )?;
    }

    // Write edges
    for to_vid in cg.g.vertices() {
        let vertex = cg.g.vertex(to_vid);
        for from_vid in vertex.uses() {
            let edge_label = format!("{:?}", vertex.typ());
            writeln!(
                writer,
                "  {} -> {} [label=\"{}\"]",
                from_vid.0, to_vid.0, edge_label
            )?;
        }
    }

    writeln!(writer, "}}")
}

pub fn format_node_label<'s, Vid: UsizeId + Debug, Rt: RuntimeType>(
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
        SingleArith(arith) => {
            format!("arith: {:?}", arith)
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
        Ntt { from, to, .. } => format!("NTT\\n{:?} -> {:?}", from, to),
        RotateIdx(_, idx) => format!("Rotate\\n({})", idx),
        Slice(_, start, end) => format!("Slice\\n[{}:{}]", start, end),
        Interpolate { .. } => String::from("Interpolate"),
        Blind(_, left, right) => format!("Blind\\n[{}:{}]", left, right),
        Array(_) => String::from("Array"),
        AssmblePoly(deg, _) => format!("AssemblePoly\\n(deg={})", deg),
        Msm { .. } => String::from("MSM"),
        HashTranscript { typ, .. } => format!("Hash\\n({:?})", typ),
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
        write_graph::<MyRuntimeType>(&cg, &mut buffer)?;

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
