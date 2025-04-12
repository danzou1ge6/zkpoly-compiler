use cytoscape_visualizer as vis;

use super::*;
use pretty_print::{format_labeled_uses, format_node_label, format_source_info, get_node_color};
use std::{fmt::Debug, io::Write};
use zkpoly_common::{arith, digraph::internal::Digraph};

pub fn write_graph<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    output_vid: VertexId,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let seq = g.dfs().add_begin(output_vid).map(|(vid, _)| vid);
    write_graph_with_optional_seq(
        g,
        writer,
        Some(output_vid),
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
        Some(output_vid),
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
        Some(output_vid),
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
        Some(seq.clone().next().unwrap()),
        seq,
        true,
        |_, _| None,
        |_, _| None,
        |_, _| None,
    )
}

pub fn write_graph_with_optional_seq<'s, Ty: Debug>(
    g: &Digraph<VertexId, partial_typed::Vertex<'s, Ty>>,
    writer: &mut impl Write,
    first_shown_vid: Option<VertexId>,
    seq: impl Iterator<Item = VertexId> + Clone,
    print_seq: bool,
    override_color: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<&'static str>,
    extra_tooltip: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<String>,
    edge_tooltip: impl Fn(VertexId, &partial_typed::Vertex<'s, Ty>) -> Option<Vec<(VertexId, String)>>,
) -> std::io::Result<()> {
    let mut builder = vis::Builder::new();

    // Write nodes
    for (i, vid) in seq.clone().enumerate() {
        let vertex = g.vertex(vid);

        if let template::VertexNode::Arith { arith, .. } = vertex.node() {
            let subgraph_vertices = arith::visualize::subgraph_vertices(
                arith,
                &format!("_{}_arith_", usize::from(vid)),
                format!("v{}", usize::from(vid)),
                &mut builder,
            );

            let mut label = format!("{}: ArithGraph", vid.0,);

            if print_seq {
                label.push_str(&format!("({})", i));
            }

            let tooltip = format!(
                "{:?}\\n@{}\\n{}",
                vertex.typ(),
                format_source_info(vertex.src()),
                extra_tooltip(vid, vertex).unwrap_or_default()
            );

            let color =
                override_color(vid, vertex).map_or_else(vis::Color::light_blue, vis::Color::new);

            let cluster_id = format!("_{}_arith_", usize::from(vid));
            builder.cluster(
                cluster_id,
                vis::Cluster::new(vis::Vertex::new(label).with_color(color).with_info(tooltip))
                    .with_children(subgraph_vertices),
            );
            continue;
        }

        let mut label = format!("{}: ", usize::from(vid)) + &format_node_label(vertex.node());

        if print_seq {
            label.push_str(&format!("({})", i));
        }

        // Get node style and color
        let (color, style) = if let Some(color) = override_color(vid, vertex) {
            (vis::Color::new(color), vis::BorderStyle::Solid)
        } else {
            let style = match vertex.node().device() {
                Device::Gpu => vis::BorderStyle::Solid,
                Device::Cpu => vis::BorderStyle::Dotted,
                Device::PreferGpu => vis::BorderStyle::Dashed,
            };
            (vis::Color::new(get_node_color(vertex.node())), style)
        };

        let tooltip = format!(
            "{:?}\\n@{}\\n{}",
            vertex.typ(),
            format_source_info(vertex.src()),
            extra_tooltip(vid, vertex).unwrap_or_default()
        );

        // Write node with attributes
        let id = format!("v{}", usize::from(vid));
        builder.vertex(
            id,
            vis::Vertex::new(label)
                .with_color(color)
                .with_border_style(style)
                .with_info(tooltip),
        );
    }

    // Write edges
    for to_vid in seq {
        let vertex = g.vertex(to_vid);

        let tooltips = edge_tooltip(to_vid, vertex);

        if let template::VertexNode::Arith { arith, .. } = vertex.node() {
            arith::visualize::subgraph_edges(
                arith,
                &format!("_{}_arith_", to_vid.0),
                format!("v{}", to_vid.0),
                |ivid| format!("v{}", ivid.0),
                tooltips,
                &mut builder,
            );
            continue;
        }

        for (from_vid, label) in format_labeled_uses(vertex.node()).into_iter() {
            let tooltip = tooltips.as_ref().map(|xs| {
                xs.iter()
                    .find(|(vid, _)| *vid == from_vid)
                    .map(|(_, tooltip)| tooltip)
                    .unwrap()
            });
            let tooltip = tooltip
                .map(|x| if x.is_empty() { None } else { Some(x) })
                .flatten();

            let from_id = format!("v{}", usize::from(from_vid));
            let to_id = format!("v{}", usize::from(to_vid));

            builder.edge(
                vis::Edge::new(from_id, to_id)
                    .with_optional_info(tooltip)
                    .with_optional_label(if label.is_empty() { None } else { Some(label) }),
            );
        }
    }

    builder
        .first_show(first_shown_vid.map(|x| format!("v{}", usize::from(x))))
        .emit("Type2", writer)
}
