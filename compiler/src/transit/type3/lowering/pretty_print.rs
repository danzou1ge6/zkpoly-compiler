use std::{collections::BTreeMap, io::Write, marker::PhantomData};

use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    devices::EventTable,
    functions::FunctionTable,
    instructions::{instruction_label, labeled_mutable_uses, labeled_uses, stream, Instruction},
};

use crate::transit::type3;

use super::{InstructionId, MultithreadChunk};

pub fn print_graph<Rt: RuntimeType>(
    mt_chunk: &MultithreadChunk,
    ftab: &FunctionTable<Rt>,
    evtab: &EventTable,
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

    writeln!(writer, "digraph multithread_chunk {{")?;

    // One node for each thread
    for (name, tid) in threads {
        writeln!(writer, "  t{} [label=<", usize::from(tid))?;
        print_thread(
            &name,
            mt_chunk.threads[tid].iter().copied(),
            mt_chunk,
            &id2t3idx,
            ftab,
            writer,
        )?;
        writeln!(writer, "  >]")?;
    }

    // Fork edges
    for (fork_to, &(fork_from, _)) in mt_chunk.forks.iter() {
        writeln!(
            writer,
            "  t{}:fork-{} -> t{}",
            usize::from(fork_from),
            usize::from(*fork_to),
            usize::from(*fork_to)
        )?;
    }

    // Record-Wait edges
    let mut record_at_thread = BTreeMap::new();
    let mut wait_at_thread = BTreeMap::new();

    for (tid, _) in mt_chunk.threads.iter_with_id() {
        for inst in mt_chunk.thread_instructions(tid) {
            match inst {
                Instruction::Record { event, .. } => {
                    record_at_thread.insert(*event, tid);
                }
                Instruction::Wait { event, .. } => {
                    wait_at_thread.insert(*event, tid);
                }
                _ => {}
            }
        }
    }

    for (evid, _) in evtab.iter_with_id() {
        writeln!(
            writer,
            "  t{}:record-{} -> t{}:wait-{} [style=dashed]",
            usize::from(*record_at_thread.get(&evid).unwrap()),
            usize::from(evid),
            usize::from(*wait_at_thread.get(&evid).unwrap()),
            usize::from(evid),
        )?;
    }

    writeln!(writer, "}}")?;

    Ok(())
}

struct RMut(VariableId);

impl From<VariableId> for RMut {
    fn from(value: VariableId) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for RMut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<span class=\"r-mut-{}\">R{}</span>",
            usize::from(self.0),
            usize::from(self.0)
        )
    }
}

struct R(VariableId);

impl From<VariableId> for R {
    fn from(value: VariableId) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for R {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<span class=\"r-{}\">R{}</span>",
            usize::from(self.0),
            usize::from(self.0)
        )
    }
}

struct LabeledVars<Rr>(Vec<(VariableId, String)>, PhantomData<Rr>);

impl<Rr> LabeledVars<Rr> {
    fn new(vars: Vec<(VariableId, String)>) -> Self {
        Self(vars, PhantomData)
    }
}

impl<Rr> std::fmt::Display for LabeledVars<Rr>
where
    Rr: std::fmt::Display + From<VariableId>,
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
                write!(f, "{}={}", label, Rr::from(*v))?;
            } else {
                write!(f, "{}", Rr::from(*v))?;
            }
        }
        Ok(())
    }
}

fn print_inst<Rt: RuntimeType>(
    inst: &Instruction,
    ftab: &FunctionTable<Rt>,
    t3idx: Option<type3::InstructionIndex>,
    writer: &mut impl Write,
    indent: bool,
) -> std::io::Result<()> {
    let port = match inst {
        Instruction::Fork { new_thread, .. } => Some(format!("fork-{}", usize::from(*new_thread))),
        Instruction::Join { thread, .. } => Some(format!("join-{}", usize::from(*thread))),
        Instruction::Record { event, .. } => Some(format!("record-{}", usize::from(*event))),
        Instruction::Wait { event, .. } => Some(format!("wait-{}", usize::from(*event))),
        _ => None,
    };
    let port_str = port.map_or_else(|| "".to_string(), |s| format!("port=\"{}\"", s));

    writeln!(writer, "  <tr>")?;
    if indent {
        writeln!(writer, "    <td></td>")?;
    }

    writeln!(
        writer,
        "    <td>{}</td>",
        t3idx.map_or_else(|| "".to_string(), |t3idx| format!("{}", usize::from(t3idx)))
    )?;
    writeln!(writer, "    <td>{}</td>", instruction_label(inst, ftab))?;

    if let Some(s) = stream(inst) {
        writeln!(writer, "    <td>{}</td>", R(s))?;
    } else {
        writeln!(writer, "    <td></td>")?;
    }

    writeln!(
        writer,
        "    <td>{}</td>",
        LabeledVars::<RMut>::new(labeled_mutable_uses(inst))
    )?;
    writeln!(
        writer,
        "    <td>{}</td>",
        LabeledVars::<R>::new(labeled_mutable_uses(inst))
    )?;

    if !indent {
        writeln!(writer, "    <td></td>")?;
    }
    writeln!(writer, "    <td {}></td>", port_str)?;
    writeln!(writer, "  </tr>")?;
    Ok(())
}

fn print_thread<Rt: RuntimeType>(
    name: &str,
    insts: impl Iterator<Item = InstructionId>,
    mt_chunk: &MultithreadChunk,
    id2t3idx: &BTreeMap<InstructionId, type3::InstructionIndex>,
    ftab: &FunctionTable<Rt>,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(writer, "<table border=\"0\">")?;

    writeln!(writer, "  <tr><td colspan=\"6\">{}</td></tr>", name)?;

    for iid in insts {
        let cell = &mt_chunk.instructions[iid];
        print_inst(&cell.inst, ftab, id2t3idx.get(&iid).cloned(), writer, false)?;

        for tail in cell.tail.iter() {
            print_inst(tail, ftab, id2t3idx.get(&iid).cloned(), writer, true)?;
        }
    }

    writeln!(writer, "</table>")?;
    Ok(())
}
