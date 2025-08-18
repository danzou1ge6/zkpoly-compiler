use cytoscape_visualizer as vis;

use super::pretty_print::{format_node_label, labeled_uses};
use super::{ArithGraph, ExprId, Operation};

pub fn subgraph_vertices<I: Copy>(
    ag: &ArithGraph<I, ExprId>,
    vertex_name_prefix: &str,
    all_output_id: String,
    builder: &mut vis::DigraphBuilder,
) -> Vec<vis::Id> {
    let mut vertices = Vec::new();

    // Write nodes
    for v in ag.g.vertices() {
        let op = &ag.g.vertex(v).op;
        let label = format_node_label(op);
        let id = format!("{}{}", vertex_name_prefix, v.0);

        vertices.push(id.clone());
        builder.vertex(id, vis::Vertex::new(label));
    }

    // Write output collection node
    vertices.push(all_output_id.clone());
    builder.vertex(all_output_id, vis::Vertex::new("AllOutputs"));

    vertices
}

pub fn subgraph_edges<I: Copy + Eq>(
    ag: &ArithGraph<I, ExprId>,
    vertex_name_prefix: &str,
    all_output_id: String,
    vid: impl Fn(I) -> String,
    edge_tooltips: Option<Vec<(I, String)>>,
    builder: &mut vis::DigraphBuilder,
) {
    for v in ag.g.vertices() {
        let op = &ag.g.vertex(v).op;
        // Write internal edges
        for (us, label) in labeled_uses(op).into_iter() {
            let from_id = format!("{}{}", vertex_name_prefix, us.0);
            let to_id = format!("{}{}", vertex_name_prefix, v.0);
            builder.edge(
                vis::Edge::new(from_id, to_id).with_optional_label(if label.is_empty() {
                    None
                } else {
                    Some(label)
                }),
            );
        }
        // Write external in-edges
        if let Operation::Input { outer_id, .. } = op {
            let tooltip = edge_tooltips.as_ref().map(|xs| {
                xs.iter()
                    .find(|(x, _)| x == outer_id)
                    .map(|(_, x)| x)
                    .unwrap()
            });
            let tooltip = tooltip
                .map(|x| if x.is_empty() { None } else { Some(x) })
                .flatten();

            let from_id = vid(*outer_id);
            let to_id = format!("{}{}", vertex_name_prefix, v.0);

            builder.edge(vis::Edge::new(from_id, to_id).with_optional_info(tooltip));
        }
    }

    for (i, &output) in ag.outputs.iter().enumerate() {
        let from_id = format!("{}{}", vertex_name_prefix, output.0);
        builder.edge(vis::Edge::new(from_id, &all_output_id).with_label(format!("{i}")));
    }
}
