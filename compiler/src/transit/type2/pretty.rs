use super::template::labelling::LabelT;
use cytoscape_visualizer::{self as vis, pretty::*};
use std::fmt::Debug;
use std::ops::DerefMut;

pub mod partial_typed {

    use zkpoly_common::digraph::internal::Digraph;

    use super::super::template;
    use super::super::{partial_typed, VertexId};
    use super::*;

    pub fn debug_graph<'s, T>(
        g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<T>>>,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()>
    where
        T: Debug + Clone,
    {
        let aux =
            template::pretty::StyledVertex(|_vid, v: &partial_typed::Vertex<'s, Option<T>>| {
                vis::Vertex::new(LabelT(v.node()).to_string()).with_optional_info(
                    v.typ()
                        .as_ref()
                        .map(|t| format!("{:?}\\n@{:?}", t, v.src())),
                )
            });

        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &aux, builder.deref_mut());
        builder.emit("Debug Type Error", std::fs::File::open(output)?)
    }
}

pub mod type_error {
    use zkpoly_common::digraph::internal::Digraph;

    use super::super::template;
    use super::super::{partial_typed, VertexId};
    use super::*;

    pub fn debug_graph<'s, T>(
        g: &Digraph<VertexId, partial_typed::Vertex<'s, Option<T>>>,
        error_at: VertexId,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()>
    where
        T: Debug + Clone,
    {
        let aux =
            template::pretty::StyledVertex(|vid, v: &partial_typed::Vertex<'s, Option<T>>| {
                vis::Vertex::new(LabelT(v.node()).to_string())
                    .with_color(if vid == error_at {
                        vis::Color::red()
                    } else {
                        vis::Color::white()
                    })
                    .with_optional_info(
                        v.typ()
                            .as_ref()
                            .map(|t| format!("{:?}\\n@{:?}", t, v.src())),
                    )
            });

        let mut builder = vis::DigraphBuilder::new();
        pretty(g, &aux, builder.deref_mut());
        builder.emit("Debug Type Error", std::fs::File::open(output)?)
    }
}

pub mod unsubgraphed_cg {

    use super::super::{
        no_subgraph::{Cg, Vertex},
        template, VertexId,
    };
    use super::*;
    use zkpoly_runtime::args::RuntimeType;

    pub fn debug_cycle<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        cycle: impl Fn(VertexId) -> bool,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let aux = template::pretty::StyledVertex(|vid, v: &Vertex<'s, Rt>| {
            vis::Vertex::new(LabelT(v.node()).to_string())
                .with_color(if cycle(vid) {
                    vis::Color::red()
                } else {
                    vis::Color::white()
                })
                .with_info(format!("{:?}\\n{:?}", v.typ(), v.src()))
        });

        let mut builder = vis::DigraphBuilder::new();
        pretty(&g.g, &aux, builder.deref_mut());
        builder.emit("Debug Non-DAG", std::fs::File::open(output)?)
    }

    pub fn debug_graph<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let aux = template::pretty::StyledVertex(|_vid, v: &Vertex<'s, Rt>| {
            vis::Vertex::new(LabelT(v.node()).to_string()).with_info(format!(
                "{:?}\\n{:?}",
                v.typ(),
                v.src()
            ))
        });

        let mut builder = vis::DigraphBuilder::new();
        pretty(&g.g, &aux, builder.deref_mut());
        builder.emit("Debug Non-DAG", std::fs::File::open(output)?)
    }
}

pub mod subgraphed_cg {

    use super::super::{
        template, VertexId, {Cg, Vertex},
    };
    use super::*;
    use zkpoly_runtime::args::RuntimeType;

    pub fn debug_cycle<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        cycle: impl Fn(VertexId) -> bool,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let aux = template::pretty::StyledVertex(|vid, v: &Vertex<'s, Rt>| {
            vis::Vertex::new(LabelT(v.node()).to_string())
                .with_color(if cycle(vid) {
                    vis::Color::red()
                } else {
                    vis::Color::white()
                })
                .with_info(format!("{:?}\\n{:?}", v.typ(), v.src()))
        });

        let mut builder = vis::DigraphBuilder::new();
        pretty(&g.g, &aux, builder.deref_mut());
        builder.emit("Debug Non-DAG", std::fs::File::open(output)?)
    }

    pub fn debug_graph<'s, Rt: RuntimeType>(
        g: &Cg<'s, Rt>,
        output: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let aux = template::pretty::StyledVertex(|_vid, v: &Vertex<'s, Rt>| {
            vis::Vertex::new(LabelT(v.node()).to_string()).with_info(format!(
                "{:?}\\n{:?}",
                v.typ(),
                v.src()
            ))
        });

        let mut builder = vis::DigraphBuilder::new();
        pretty(&g.g, &aux, builder.deref_mut());
        builder.emit("Debug Non-DAG", std::fs::File::open(output)?)
    }
}
