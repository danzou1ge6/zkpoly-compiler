use cytoscape_visualizer::*;

fn main() {
    // Take first CLI argument
    let dst_fname = std::env::args().nth(1).unwrap();

    let mut f = std::fs::File::create(dst_fname).unwrap();

    let mut digraph = DigraphBuilder::new();
    digraph
        .vertex(
            "v1",
            Vertex::new("v1").with_border_style(BorderStyle::Dashed),
        )
        .vertex("v3", Vertex::new("v3"))
        .vertex("v4", Vertex::new("v4").with_color(Color::light_red()))
        .vertex("v5", Vertex::new("v5"))
        .cluster("v2", {
            let mut cluster =
                ClusterBuilder::new(Vertex::new("v2").with_color(Color::light_green()));
            cluster
                .vertex("v2_1", Vertex::new("v2_2").with_color(Color::light_blue()))
                .vertex("v2_2", Vertex::new("v2_2"));
            cluster
        })
        .edge(
            Edge::new("v2_1", "v1")
                .with_label("e1")
                .with_info("tooltip test"),
        )
        .edge_plain("v2_2", "v2_1")
        .edge(Edge::new("v3", "v2_1").with_label("e2"))
        .edge_plain("v4", "v2_2")
        .edge_plain("v4", "v3")
        .edge_plain("v5", "v4");

    digraph
        .first_show(["v1"])
        .emit("Simple Graph", &mut f)
        .unwrap();
}
