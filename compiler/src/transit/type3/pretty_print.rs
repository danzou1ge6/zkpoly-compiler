use std::io::Write;

use super::*;
use crate::transit::type2;
use zkpoly_runtime::args::RuntimeType;

pub fn prettify<'s, Rt: RuntimeType>(
    chunk: &Chunk<'s, Rt>,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let head = r#"
    <table>
        <thead>
            <tr>
            <th>Number</th>
            <th colspan=3>Definitions</th>
            <th>Instruction</th>
            <th>Uses</th>
            <th>Source</th>
            </tr>
        </thead>
        <tbody>
    "#;
    writeln!(writer, "{}", head)?;

    chunk
        .iter_instructions()
        .map(|(idx, inst)| prettify_inst(chunk, idx, inst, writer))
        .collect::<Result<Vec<_>, _>>()?;

    let tail = r#"
        </tbody>
    </table>
    <script>
        function extractNumber(classNames) {
            const regex = /^register-(use|def)-(\d+)$/;
            for (const className of classNames) {
                const match = className.match(regex);
                if (match) {
                    return parseInt(match[2], 10);
                }
            }
            return null;
        }
        function setStyle(regNum) {
            let all_regs = document.querySelectorAll('.register')
            let all_rows = document.querySelectorAll('tr')
            let uses = document.querySelectorAll('.register-use-' + regNum)
            let defs = document.querySelectorAll('.register-def-' + regNum)
            let use_rows = document.querySelectorAll('.row-use-' + regNum)
            let def_rows = document.querySelectorAll('.row-def-' + regNum)

            all_regs.forEach(r => r.style.background = 'white')
            uses.forEach(use => use.style.background = "rgb(173, 216, 230)")
            defs.forEach(def => def.style.background = "rgb(144, 238, 144)")

            all_rows.forEach(r => r.style.color = 'gray')
            use_rows.forEach(r => r.style.color = 'black')
            def_rows.forEach(r => r.style.color = 'black')
        }

        document.querySelectorAll('.register').forEach(r => {
            let n = extractNumber(r.classList)
            r.addEventListener('click', ev => setStyle(n))
        })
    </script>
    "#;
    writeln!(writer, "{}", tail)?;

    Ok(())
}

fn reg_id2str_def(r: RegisterId) -> String {
    format!(
        "<span class=\"register register-def-{}\">r{}</span>",
        r.0, r.0
    )
}

fn reg_id2str_use(r: RegisterId) -> String {
    format!(
        "<span class=\"register register-use-{}\">r{}</span>",
        r.0, r.0
    )
}

fn typ2str(t: &super::typ::Typ) -> String {
    format!("{:?}", t)
}

fn prettify_inst<'s, Rt: RuntimeType>(
    chunk: &Chunk<'s, Rt>,
    idx: InstructionIndex,
    inst: &Instruction<'s>,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let def_rows: Vec<_> = inst
        .defs()
        .map(|def| {
            let typ = &chunk.register_types[def];
            let dev = chunk
                .register_devices
                .get(&def)
                .map_or_else(|| "ERROR".to_string(), |dev| format!("{:?}", dev));
            vec![reg_id2str_def(def), typ2str(typ), dev]
        })
        .collect();

    let labeled_uses = format_labeled_uses(inst);

    let classes: Vec<_> = inst
        .defs()
        .map(|def| format!("row-def-{}", def.0))
        .chain(labeled_uses.iter().map(|(r, _)| format!("row-use-{}", r.0)))
        .collect();
    let classes_str = classes.join(" ");

    writeln!(writer, "<tr class=\"{}\">", classes_str)?;
    let index = format!("{}", idx.0);
    writeln!(writer, "  <th rowspan={}>{}</th>", def_rows.len(), index)?;

    if let Some(first_def_row) = def_rows.get(0) {
        for def in first_def_row {
            writeln!(writer, "  <td>{}</td>", def)?;
        }
    } else {
        for _ in 0..3 {
            writeln!(writer, "  <td></td>")?;
        }
    }

    let label = format_inst_label(inst, chunk);
    let uses_str = labeled_uses
        .into_iter()
        .map(|(id, label)| {
            if label.is_empty() {
                reg_id2str_use(id)
            } else {
                format!("{}={}", label, reg_id2str_use(id))
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let src_info = format_src_info(inst);

    writeln!(writer, "  <td>{}</td>", label)?;
    writeln!(writer, "  <td>{}</td>", uses_str)?;
    writeln!(writer, "  <td>{}</td>", src_info)?;
    writeln!(writer, "</tr>")?;

    for def in def_rows.iter().skip(1) {
        writeln!(writer, "<tr class=\"{}\">", classes_str)?;

        for def in def {
            writeln!(writer, "  <td>{}</td>", def)?;
        }

        writeln!(writer, "</tr>")?;
    }

    Ok(())
}

fn format_src_info(inst: &Instruction) -> String {
    if let Some(src) = &inst.src {
        type2::pretty_print::format_source_info(&src)
    } else {
        "".to_string()
    }
}

fn format_inst_label<'s, Rt: RuntimeType>(inst: &Instruction<'s>, chunk: &Chunk<'s, Rt>) -> String {
    use template::InstructionNode::*;
    match &inst.node {
        Type2 { vertex, .. } => type2::pretty_print::format_node_label(vertex),
        GpuMalloc { addr: aid, .. } => {
            let (addr, size) = chunk.gpu_addr_mapping[*aid];
            format!("GpuMalloc({:?}={},{})", aid, addr.0, size)
        }
        GpuFree { .. } => "GpuFree".to_string(),
        CpuMalloc { .. } => "CpuMalloc".to_string(),
        CpuFree { .. } => "CpuFree".to_string(),
        StackFree { .. } => "StackFree".to_string(),
        Tuple { .. } => "Tuple".to_string(),
        Transfer { .. } => "Transfer".to_string(),
        Move { .. } => "Move".to_string(),
        SetPolyMeta { offset, len, .. } => format!("SetPolyMeta({}, {})", offset, len),
    }
}

fn format_labeled_uses<'s>(inst: &Instruction<'s>) -> Vec<(RegisterId, String)> {
    use template::InstructionNode::*;
    match &inst.node {
        Type2 { vertex, temp, .. } => type2::pretty_print::format_labeled_uses(vertex)
            .into_iter()
            .chain(
                temp.iter()
                    .enumerate()
                    .map(|(i, t)| (*t, format!("temp{}", i))),
            )
            .collect(),
        GpuMalloc { .. } => vec![],
        GpuFree { id } => vec![(*id, "".to_string())],
        CpuMalloc { .. } => vec![],
        CpuFree { id } => vec![(*id, "".to_string())],
        StackFree { id } => vec![(*id, "".to_string())],
        Tuple { oprands, .. } => oprands
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, "".to_string()))
            .collect(),
        Transfer { from, .. } => vec![(*from, "".to_string())],
        Move { from, .. } => vec![(*from, "".to_string())],
        SetPolyMeta { from, .. } => vec![(*from, "".to_string())],
    }
}
