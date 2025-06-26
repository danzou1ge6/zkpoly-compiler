use std::{collections::BTreeMap, io::Write, marker::PhantomData};

use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    functions::FunctionTable,
    instructions::{
        instruction_label, labeled_mutable_uses, labeled_uses, static_args, stream, Instruction, InstructionNode,
    },
};

use crate::transit::type3;

use super::{InstructionId, MultithreadChunk, Stream, StreamSpecific};

pub fn print<Rt: RuntimeType>(
    mt_chunk: &MultithreadChunk,
    stream2variable_id: &StreamSpecific<VariableId>,
    ftab: &FunctionTable<Rt>,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let id2t3idx: BTreeMap<_, _> = mt_chunk
        .t3idx2id
        .iter()
        .map(|(id, t3idx)| (*t3idx, *id))
        .collect();
    let threads = mt_chunk
        .primary_thread_id
        .iter()
        .map(|(pt, tid)| ((format!("{:?}", pt), *tid)))
        .chain(
            mt_chunk
                .threads
                .iter_with_id()
                .filter(|(tid, _)| {
                    mt_chunk
                        .primary_thread_id
                        .iter()
                        .all(|(_, ptid)| ptid != tid)
                })
                .map(|(tid, _)| (format!("{}", usize::from(tid)), tid)),
        );
    let variable_id2stream: BTreeMap<_, _> =
        stream2variable_id.iter().map(|(v, s)| (*s, v)).collect();

    writeln!(writer, "<div>")?;

    // One table for each thread
    for (name, tid) in threads {
        writeln!(writer, "<div id=\"thread-{}\">", usize::from(tid))?;
        writeln!(writer, "<h2>{}</h2>", name)?;
        let head = r#"
        <table>
            <thead>
                <tr>
                <th>Type3 Inst.</th>
                <th>Label</th>
                <th>Stream</th>
                <th>Mutable Uses</th>
                <th>Uses</th>
                </tr>
            </thead>
        <tbody>
        "#;

        writeln!(writer, "{}", head)?;

        print_thread(
            mt_chunk.threads[tid].iter().copied(),
            mt_chunk,
            &id2t3idx,
            ftab,
            &variable_id2stream,
            writer,
        )?;

        writeln!(writer, "</tbody>\n</table>")?;
        writeln!(writer, "</div>")?;
    }

    writeln!(writer, "</div>")?;

    let script = r#"
    <script>
        function extractNumber(classNames) {
            const regex = /^register-(mut-)?(\d+)$/;
            for (const className of classNames) {
                const match = className.match(regex);
                if (match) {
                    return parseInt(match[2], 10);
                }
            }
            return null;
        }
        function setRegStyle(regNum) {
            let all_regs = document.querySelectorAll('.register')
            let all_rows = document.querySelectorAll('tr')
            let uses = document.querySelectorAll('.register-' + regNum)
            let mutable_uses = document.querySelectorAll('.register-mut-' + regNum)
            let use_rows = document.querySelectorAll('.row-use-' + regNum)

            all_regs.forEach(r => r.style.background = 'white')
            uses.forEach(use => use.style.background = "rgb(173, 216, 230)")
            mutable_uses.forEach(def => def.style.background = "rgb(144, 238, 144)")

            all_rows.forEach(r => r.style.color = 'rgb(125, 125, 125)')
            use_rows.forEach(r => r.style.color = 'black')
        }
        function isElementVisible(el) {
            const rect = el.getBoundingClientRect();
            const windowHeight = window.innerHeight || document.documentElement.clientHeight;
            const windowWidth = window.innerWidth || document.documentElement.clientWidth;
            
            const visibilityThreshold = rect.height * 0.3;
            
            const verticalVisible = (
                rect.top <= windowHeight - visibilityThreshold &&
                rect.bottom >= visibilityThreshold
            );

            const horizontalVisible = (
                rect.left <= windowWidth &&
                rect.right >= 0
            );

            return verticalVisible && horizontalVisible;
        }

        document.querySelectorAll('.register').forEach(r => {
            let n = extractNumber(r.classList)
            r.addEventListener('click', ev => setRegStyle(n))
        })

        document.querySelectorAll('.row-fork, .row-record, .row-wait').forEach(element => {
            element.addEventListener('click', function() {
                let targetPrefix
                if (this.classList.contains('row-fork')) {
                    targetPrefix = 'thread'
                } else if (this.classList.contains('row-record')) {
                    targetPrefix = 'row-wait'
                } else if (this.classList.contains('row-wait')) {
                    targetPrefix = 'row-record'
                }

                const idParts = this.id.split('-')
                const n = idParts[2]

                const targetId = `${targetPrefix}-${n}`
                let target = document.getElementById(targetId)

                target.style.background = 'rgb(251, 255, 0)'
                setTimeout(() => {
                    target.style.background = 'white'
                }, 2000)
                if (!isElementVisible(target))
                    target.scrollIntoView({ behavior: 'smooth' })
            });
        });
    </script>
    "#;

    writeln!(writer, "{}", script)?;

    Ok(())
}

type V2S = BTreeMap<VariableId, Stream>;

struct RMut<'a>(VariableId, &'a V2S);

impl<'a> From<(VariableId, &'a V2S)> for RMut<'a> {
    fn from((v, s): (VariableId, &'a V2S)) -> Self {
        Self(v, s)
    }
}

impl<'a> std::fmt::Display for RMut<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let usize_id = usize::from(self.0);
        let stream_str = self
            .1
            .get(&self.0)
            .map_or_else(|| String::new(), |s| format!("({:?})", s));
        write!(
            f,
            "<span class=\"register register-mut-{}\">r{}{}</span>",
            usize_id, usize_id, stream_str
        )
    }
}

struct R<'a>(VariableId, &'a V2S);

impl<'a> From<(VariableId, &'a V2S)> for R<'a> {
    fn from((v, s): (VariableId, &'a V2S)) -> Self {
        Self(v, s)
    }
}

impl<'a> std::fmt::Display for R<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let usize_id = usize::from(self.0);
        let stream_str = self
            .1
            .get(&self.0)
            .map_or_else(|| String::new(), |s| format!("({:?})", s));
        write!(
            f,
            "<span class=\"register register-{}\">r{}{}</span>",
            usize_id, usize_id, stream_str
        )
    }
}

struct LabeledVars<'a, Rr>(Vec<(VariableId, String)>, &'a V2S, PhantomData<Rr>);

impl<'a, Rr> LabeledVars<'a, Rr> {
    fn new(vars: Vec<(VariableId, String)>, v2s: &'a V2S) -> Self {
        Self(vars, v2s, PhantomData)
    }
}

impl<'a, Rr> std::fmt::Display for LabeledVars<'a, Rr>
where
    Rr: std::fmt::Display + From<(VariableId, &'a V2S)>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for ((v, label), first) in self
            .0
            .iter()
            .zip(std::iter::once(true).chain(std::iter::repeat(false)))
        {
            if !first {
                write!(f, ", ")?;
            }
            if label != "" {
                write!(f, "{}={}", label, Rr::from((*v, self.1)))?;
            } else {
                write!(f, "{}", Rr::from((*v, self.1)))?;
            }
        }
        Ok(())
    }
}

fn print_inst<Rt: RuntimeType>(
    inst: &Instruction,
    ftab: &FunctionTable<Rt>,
    t3idx: Option<type3::InstructionIndex>,
    v2s: &V2S,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let id = match inst.node() {
        InstructionNode::Fork { new_thread, .. } => {
            Some((format!("row-fork-{}", usize::from(*new_thread)), "row-fork"))
        }
        InstructionNode::Join { thread, .. } => {
            Some((format!("row-join-{}", usize::from(*thread)), "row-join"))
        }
        InstructionNode::Record { event, .. } => {
            Some((format!("row-record-{}", usize::from(*event)), "row-record"))
        }
        InstructionNode::Wait { event, .. } => {
            Some((format!("row-wait-{}", usize::from(*event)), "row-wait"))
        }
        _ => None,
    };

    let mutable_uses = labeled_mutable_uses(inst.node());
    let uses = labeled_uses(inst.node());

    let classes: Vec<_> = id
        .as_ref()
        .map(|(_, c)| c.to_string())
        .into_iter()
        .chain(
            mutable_uses
                .iter()
                .map(|(v, _)| format!("row-use-{}", usize::from(*v))),
        )
        .chain(
            uses.iter()
                .map(|(v, _)| format!("row-use-{}", usize::from(*v))),
        )
        .collect();

    let id_str = id.map_or_else(|| String::new(), |(id, _)| format!(" id=\"{}\"", id));
    let class_str = if !classes.is_empty() {
        format!(" class=\"{}\"", classes.join(" "))
    } else {
        "".to_string()
    };

    writeln!(writer, "  <tr{}{}>", id_str, class_str)?;

    writeln!(
        writer,
        "    <td>{}</td>",
        t3idx.map_or_else(|| "".to_string(), |t3idx| format!("{}", usize::from(t3idx)))
    )?;
    writeln!(writer, "    <td>{}</td>", instruction_label(inst.node(), ftab))?;

    if let Some(s) = stream(inst.node()) {
        writeln!(writer, "    <td>{}</td>", R(s, v2s))?;
    } else {
        writeln!(writer, "    <td></td>")?;
    }

    writeln!(
        writer,
        "    <td>{}</td>",
        LabeledVars::<RMut>::new(mutable_uses, v2s)
    )?;

    if let Some(args) = static_args(inst.node()) {
        writeln!(writer, "    <td>{}</td>", args)?;
    } else {
        writeln!(writer, "    <td>{}</td>", LabeledVars::<R>::new(uses, v2s))?;
    }

    writeln!(writer, "  </tr>")?;
    Ok(())
}

fn print_thread<Rt: RuntimeType>(
    insts: impl Iterator<Item = InstructionId>,
    mt_chunk: &MultithreadChunk,
    id2t3idx: &BTreeMap<InstructionId, type3::InstructionIndex>,
    ftab: &FunctionTable<Rt>,
    v2s: &V2S,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    for iid in insts {
        let cell = &mt_chunk.instructions[iid];
        print_inst(&cell.inst, ftab, id2t3idx.get(&iid).cloned(), v2s, writer)?;

        for tail in cell.tail.iter() {
            print_inst(tail, ftab, id2t3idx.get(&iid).cloned(), v2s, writer)?;
        }
    }
    Ok(())
}
