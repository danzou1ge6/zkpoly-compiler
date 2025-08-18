use super::{DigraphBuilder, Edge, Vertex};
use std::io;

fn detach_thread<T>(jh: std::thread::JoinHandle<T>) {
    use std::os::unix::thread::JoinHandleExt;
    let pthread = jh.into_pthread_t();
    unsafe {
        let r = libc::pthread_detach(pthread);
        if r != 0 {
            panic!("pthread_detach returned non-zero code {}", r)
        }
    }
}

impl DigraphBuilder {
    pub fn emit_graphviz(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Write the DOT header
        writeln!(f, "digraph ComputationGraph {{")?;
        writeln!(f, "  // Graph settings")?;
        writeln!(f, "  graph [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "  ]")?;

        // Node settings
        writeln!(f, "  // Node settings")?;
        writeln!(f, "  node [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "    shape = \"box\"")?;
        writeln!(f, "    style = \"rounded\"")?;
        writeln!(f, "  ]")?;

        // Edge settings
        writeln!(f, "  // Edge settings")?;
        writeln!(f, "  edge [")?;
        writeln!(f, "    fontname = \"Helvetica\"")?;
        writeln!(f, "    fontsize = 11")?;
        writeln!(f, "  ]")?;

        Ok(())
    }
}
