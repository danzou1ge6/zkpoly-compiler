use std::io::{self, Write};

pub static COLORS: [&str; 8] = [
    "#4facfe", "#a6c0fe", "#f093fb", "#f5576c", "#5ee7df", "#38ef7d", "#f6d365", "#a8edea",
];

static TEMPLATE: &'static str = include_str!("renderer.html");

pub fn color_loop() -> impl Iterator<Item = &'static str> {
    COLORS.iter().cycle().copied()
}

#[derive(Debug, Clone)]
pub struct Category {
    name: String,
    color: String,
}

impl Category {
    pub fn new(name: String, color: String) -> Category {
        Category { name, color }
    }

    fn build(&self, id: usize, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        write!(
            writer,
            "    {{ id: {}, name: \"{}\", color: \"{}\" }}",
            id, self.name, self.color
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub id: String,
    pub label: String,
    pub start: u128,
    pub end: u128,
    pub success: bool,
    pub worker: String,
}

#[derive(Debug, Clone)]
pub struct Builder {
    title: String,
    categories: Vec<Category>,
    data: Vec<Vec<Entry>>,
}

fn replace_and_write_to<'a, W: Write>(
    f: &mut W,
    template: &str,
    replacements: Vec<(&str, Box<dyn FnOnce(&mut W) -> std::io::Result<()> + 'a>)>,
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

impl Builder {
    pub fn new(title: String) -> Self {
        Builder {
            title,
            categories: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn add_category(&mut self, category: Category) -> usize {
        let id = self.categories.len();
        self.categories.push(category);
        self.data.push(Vec::new());
        id
    }

    pub fn add_entry(&mut self, category_id: usize, entry: Entry) {
        if self.categories.len() <= category_id {
            panic!("category_id out of range");
        }

        self.data[category_id].push(entry);
    }

    fn build_categories(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "const TASK_TYPES = [")?;

        for (i, category) in self.categories.iter().enumerate() {
            category.build(i, writer)?;
            writeln!(writer, ",")?;
        }

        writeln!(writer, "];")?;

        Ok(())
    }

    fn build_category_data(
        &self,
        i: usize,
        writer: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        writeln!(writer, "[")?;
        for entry in self.data[i].iter() {
            let status = if entry.success { "success" } else { "fail" };

            writeln!(
                writer,
                "    {{ id: \"{}\", typeId: {}, name: \"{}\", start: {}, end: {}, color: \"{}\", metadata: {{ status: \"{}\", duration: \"{}\", worker: \"{}\" }} }},",
                entry.id,
                i,
                entry.label,
                entry.start,
                entry.end,
                self.categories[i].color,
                status,
                entry.end - entry.start,
                entry.worker
            )?;
        }

        writeln!(writer, "]")?;

        Ok(())
    }

    fn build_data(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "const taskGroups = [")?;
        for (i, _) in self.data.iter().enumerate() {
            write!(writer, "{{ type: ")?;
            self.categories[i].build(i, writer)?;
            write!(writer, ", tasks: ")?;
            self.build_category_data(i, writer)?;
            writeln!(writer, "}}, ")?;
        }

        writeln!(writer, "];")?;
        Ok(())
    }

    pub fn build(self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        replace_and_write_to(
            writer,
            TEMPLATE,
            vec![
                ("{{title}}", Box::new(|f| write!(f, "{}", self.title))),
                ("{{categories}}", Box::new(|f| self.build_categories(f))),
                ("{{data}}", Box::new(|f| self.build_data(f))),
            ],
        )
    }
}
