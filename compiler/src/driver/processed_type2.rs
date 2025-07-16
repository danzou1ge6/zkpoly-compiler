use crate::ast;
use std::io::Write;

use super::ConstantPool;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::{self, RuntimeType};

use super::type2;

pub struct ProcessedType2<'s, Rt: RuntimeType> {
    pub(super) cg: type2::unsliced::Cg<'s, Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
}

impl<'s, Rt: RuntimeType> ProcessedType2<'s, Rt> {
    pub fn slice(self) {
        let _cg = type2::subgraph_slicing::fuse(self.cg, 20);
    }
}
