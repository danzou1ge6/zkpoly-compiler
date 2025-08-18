use std::{
    borrow::Borrow,
    collections::HashMap,
    io::{self, Write},
};

mod graphviz;
pub mod pretty;

static TEMPLATE: &'static str = include_str!("template.html");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color(&'static str);

impl Color {
    pub fn new(hex: &'static str) -> Self {
        Color(hex)
    }

    pub fn white() -> Self {
        Color("#ffffff")
    }

    pub fn black() -> Self {
        Color("#000000")
    }

    pub fn gray() -> Self {
        Color("#808080")
    }

    pub fn light_green() -> Self {
        Color("#90ee90")
    }

    pub fn light_blue() -> Self {
        Color("#add8e6")
    }

    pub fn orange() -> Self {
        Color("#ffa500")
    }

    pub fn red() -> Self {
        Color("#ff0000")
    }

    pub fn light_red() -> Self {
        Color("#ff9999")
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::white()
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderStyle {
    Solid,
    Dashed,
    Dotted,
}

impl std::fmt::Display for BorderStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BorderStyle::Solid => write!(f, "solid"),
            BorderStyle::Dashed => write!(f, "dashed"),
            BorderStyle::Dotted => write!(f, "dotted"),
        }
    }
}

impl Default for BorderStyle {
    fn default() -> Self {
        BorderStyle::Solid
    }
}

pub type Id = String;

#[derive(Debug, Clone)]
pub struct Vertex {
    label: String,
    color: Color,
    parent: Option<Id>,
    border_style: BorderStyle,
    incoming: Vec<Id>,
    info: Option<String>,
}

impl Vertex {
    pub fn new(label: impl Into<String>) -> Self {
        Vertex {
            label: label.into(),
            color: Color::default(),
            parent: None,
            border_style: BorderStyle::default(),
            incoming: Vec::new(),
            info: None,
        }
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_border_style(mut self, border_style: BorderStyle) -> Self {
        self.border_style = border_style;
        self
    }

    pub fn with_info(mut self, info: String) -> Self {
        self.info = Some(info);
        self
    }

    pub fn with_optional_info(mut self, info: Option<String>) -> Self {
        self.info = info;
        self
    }

    fn with_parent(self, parent: Id) -> Self {
        Self {
            parent: Some(parent),
            ..self
        }
    }
}

impl Vertex {
    fn emit(&self, f: &mut impl Write, in_cluster: bool) -> io::Result<()> {
        write!(f, "label: \"{}\"", self.label)?;
        if let Some(parent) = &self.parent {
            write!(f, ", parent: \"{}\"", parent)?;
        }
        if in_cluster {
            write!(f, ", classes: [\"cluster\"]")?;
        } else {
            write!(f, ", classes: [\"vertex\"]")?;
        }
        write!(f, ", color: \"{}\"", self.color)?;
        write!(f, ", borderWidth: \"{}\"", self.border_style)?;

        write!(f, ", incoming: [")?;
        for (i, incoming) in self.incoming.iter().enumerate() {
            write!(f, "\"{}\"", incoming)?;
            if i != self.incoming.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;

        if let Some(info) = &self.info {
            write!(f, ", info: \"{}\"", info)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Cluster {
    v: Vertex,
    children: Vec<Id>,
}

impl Cluster {
    fn emit(&self, f: &mut impl Write) -> io::Result<()> {
        self.v.emit(f, true)?;
        write!(f, ", children: [")?;
        for (i, child) in self.children.iter().enumerate() {
            write!(f, "\"{}\"", child)?;
            if i != self.children.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ClusterBuilder {
    v: Vertex,
    builder: Builder,
}

impl std::ops::Deref for ClusterBuilder {
    type Target = Builder;
    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl std::ops::DerefMut for ClusterBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

impl ClusterBuilder {
    pub fn new(v_config: Vertex) -> Self {
        ClusterBuilder {
            v: v_config,
            builder: Builder::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DigraphBuilder {
    builder: Builder,
    first_shown_vertices: Vec<Id>,
}

impl std::ops::Deref for DigraphBuilder {
    type Target = Builder;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl std::ops::DerefMut for DigraphBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

impl DigraphBuilder {
    pub fn new() -> Self {
        DigraphBuilder {
            builder: Builder::new(),
            first_shown_vertices: Vec::new(),
        }
    }

    pub fn first_show(&mut self, ids: impl IntoIterator<Item = impl Into<Id>>) -> &mut Self {
        ids.into_iter().for_each(|i| {
            self.first_shown_vertices.push(i.into());
        });
        self
    }
}

#[derive(Default, Clone, Debug)]
pub struct EdgeStyle {
    label: Option<String>,
    info: Option<String>,
}

impl EdgeStyle {
    pub fn with_optional_label(mut self, label: Option<impl Into<String>>) -> Self {
        self.label = label.map(Into::into);
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_optional_info(mut self, info: Option<impl Into<String>>) -> Self {
        self.info = info.map(Into::into);
        self
    }

    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.info = Some(info.into());
        self
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    from: Id,
    to: Id,
    style: EdgeStyle,
}

impl Edge {
    pub fn new(from: impl Into<Id>, to: impl Into<Id>) -> Self {
        Edge {
            from: from.into(),
            to: to.into(),
            style: EdgeStyle::default(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.style.label = Some(label.into());
        self
    }

    pub fn with_optional_label(mut self, label: Option<impl Into<String>>) -> Self {
        self.style.label = label.map(Into::into);
        self
    }

    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.style.info = Some(info.into());
        self
    }

    pub fn with_optional_info(mut self, info: Option<impl Into<String>>) -> Self {
        self.style.info = info.map(Into::into);
        self
    }

    pub fn with_style(self, style: EdgeStyle) -> Self {
        Self { style, ..self }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    vertices: HashMap<Id, Vertex>,
    clusters: HashMap<Id, Cluster>,
    edges_inverse: HashMap<Id, Vec<Id>>,
    edge_labels: HashMap<Id, HashMap<Id, String>>,
    edge_infos: HashMap<Id, HashMap<Id, String>>,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            vertices: HashMap::new(),
            clusters: HashMap::new(),
            edges_inverse: HashMap::new(),
            edge_labels: HashMap::new(),
            edge_infos: HashMap::new(),
        }
    }

    pub fn vertex(&mut self, id: impl Into<Id> + Clone, v: Vertex) -> &mut Self {
        self.vertices.insert(id.clone().into(), v);
        self.edges_inverse.insert(id.into(), Vec::new());
        self
    }

    pub fn cluster(&mut self, id: impl Into<Id> + Clone, cluster: ClusterBuilder) -> &mut Self {
        self.clusters.insert(
            id.clone().into(),
            Cluster {
                v: cluster.v,
                children: cluster.builder.vertices.keys().cloned().collect(),
            },
        );
        self.vertices.extend(
            cluster
                .builder
                .vertices
                .into_iter()
                .map(|(i, v)| (i, v.with_parent(id.clone().into()))),
        );
        self.clusters.extend(cluster.builder.clusters);
        self.edges_inverse.insert(id.into(), Vec::new());
        cluster
            .builder
            .edges_inverse
            .into_iter()
            .for_each(|(from, to)| self.edges_inverse.entry(from).or_default().extend(to));
        cluster
            .builder
            .edge_labels
            .into_iter()
            .for_each(|(to, from_labels)| {
                self.edge_labels.entry(to).or_default().extend(from_labels)
            });
        cluster
            .builder
            .edge_infos
            .into_iter()
            .for_each(|(to, from_infos)| {
                self.edge_labels.entry(to).or_default().extend(from_infos)
            });
        self
    }

    pub fn edge(&mut self, edge: Edge) -> &mut Self {
        if let Some(label) = edge.style.label {
            self.edge_labels
                .entry(edge.to.clone().into())
                .or_insert(HashMap::new())
                .insert(edge.from.clone().into(), label);
        }

        if let Some(info) = edge.style.info {
            self.edge_infos
                .entry(edge.to.clone().into())
                .or_insert(HashMap::new())
                .insert(edge.from.clone().into(), info);
        }

        self.edge_plain(edge.from, edge.to)
    }

    pub fn edge_plain(
        &mut self,
        from: impl Into<Id> + Borrow<str>,
        to: impl Into<Id> + Borrow<str>,
    ) -> &mut Self {
        let points_to = self
            .edges_inverse
            .get_mut(from.borrow())
            .unwrap_or_else(|| panic!("Vertex with ID {} not created yet", from.borrow()));
        self.vertices
            .get_mut(to.borrow())
            .unwrap()
            .incoming
            .push(from.into());
        points_to.push(to.into());
        self
    }
}

fn replace_and_write_to<'a, W: Write>(
    f: &mut W,
    template: &str,
    replacements: Vec<(&str, Box<dyn FnOnce(&mut W) -> io::Result<()> + 'a>)>,
) -> io::Result<()> {
    let mut template_parts = vec![template];

    for (k, _) in replacements.iter() {
        let s = template_parts.pop().unwrap();
        let (s1, s2) = s.split_once(k).unwrap();

        template_parts.push(s1);
        template_parts.push(s2);
    }

    for (part, (_, v)) in template_parts.iter().zip(replacements.into_iter()) {
        write!(f, "{}", part)?;
        v(f)?;
    }

    write!(f, "{}", template_parts.pop().unwrap())?;

    Ok(())
}

impl DigraphBuilder {
    fn emit_vertices(&self, f: &mut impl Write) -> io::Result<()> {
        for (id, v) in self.builder.vertices.iter() {
            write!(f, "  \"{}\": {{", id)?;
            v.emit(f, false)?;
            writeln!(f, "}},")?;
        }

        for (id, cluster) in self.builder.clusters.iter() {
            write!(f, "  \"{}\": {{", id)?;
            cluster.emit(f)?;
            writeln!(f, "}},")?;
        }

        Ok(())
    }

    fn emit_edges_inv(&self, f: &mut impl Write) -> io::Result<()> {
        for (id, outgoings) in self.builder.edges_inverse.iter() {
            write!(f, "  \"{}\": [", id)?;
            for (i, oid) in outgoings.iter().enumerate() {
                write!(f, "\"{}\"", oid)?;
                if i != outgoings.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "],")?;
        }

        Ok(())
    }

    fn emit_edge_labels(&self, f: &mut impl Write) -> io::Result<()> {
        for (from, to_labels) in self.builder.edge_labels.iter() {
            write!(f, "  \"{}\": {{", from)?;

            for (to, label) in to_labels.iter() {
                write!(f, "\"{}\": \"{}\"", to, label)?;
                if to != to_labels.iter().last().unwrap().0 {
                    write!(f, ", ")?;
                }
            }

            writeln!(f, "}},")?;
        }

        Ok(())
    }

    fn emit_edge_infos(&self, f: &mut impl Write) -> io::Result<()> {
        for (from, to_infos) in self.builder.edge_infos.iter() {
            write!(f, "  \"{}\": {{", from)?;

            for (to, info) in to_infos.iter() {
                write!(f, "\"{}\": \"{}\"", to, info)?;
                if to != to_infos.iter().last().unwrap().0 {
                    write!(f, ", ")?;
                }
            }

            writeln!(f, "}},")?;
        }

        Ok(())
    }

    fn emit_all(&self, f: &mut impl Write) -> io::Result<()> {
        writeln!(f, "const vertices = {{")?;
        self.emit_vertices(f)?;
        writeln!(f, "}}")?;

        writeln!(f, "const edgesInv = {{")?;
        self.emit_edges_inv(f)?;
        writeln!(f, "}}")?;

        writeln!(f, "const edgeLabels = {{")?;
        self.emit_edge_labels(f)?;
        writeln!(f, "}}")?;

        writeln!(f, "const edgeInfos = {{")?;
        self.emit_edge_infos(f)?;
        writeln!(f, "}}")?;

        write!(f, "const firstShownVertices = [")?;
        for (i, id) in self.first_shown_vertices.iter().enumerate() {
            write!(f, "\"{}\"", id)?;
            if i != self.first_shown_vertices.len() - 1 {
                write!(f, ", ")?;
            }
        }
        writeln!(f, "]")?;

        Ok(())
    }

    pub fn emit(&self, title: &str, mut f: impl Write) -> io::Result<()> {
        replace_and_write_to(
            &mut f,
            TEMPLATE,
            vec![
                ("{{title}}", Box::new(|f| write!(f, "{}", title))),
                ("{{data}}", Box::new(|f| self.emit_all(f))),
            ],
        )
    }
}
