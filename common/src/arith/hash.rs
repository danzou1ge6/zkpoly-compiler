use std::collections::{BTreeMap, VecDeque};

use super::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Id {
    Input(usize),
    Middle(usize, usize),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Input {
    typ: FusedType,
    mutability: Mutability,
    duplicat: i32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Output {
    typ: FusedType,
    store_node: Id,
    in_node: Vec<Id>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Tail {
    Middle(Arith<Id>),
    Output(Output),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct NormalizedDag {
    inputs: Vec<Input>,
    middles: Vec<Vec<Tail>>,
}

impl<Oid, Aid> From<&ArithGraph<Oid, Aid>> for NormalizedDag
where
    Oid: UsizeId + Ord,
    Aid: UsizeId + 'static,
{
    fn from(ag: &ArithGraph<Oid, Aid>) -> Self {
        let mut input_used = BTreeMap::new();
        let inputs = ag
            .inputs
            .iter()
            .enumerate()
            .map(|(id, i)| match &ag.g.vertex(*i).op {
                Operation::Input {
                    typ,
                    mutability,
                    outer_id,
                } => {
                    if input_used.contains_key(outer_id) {
                        let duplicate = input_used[outer_id];
                        (
                            *i,
                            Input {
                                typ: typ.clone(),
                                mutability: *mutability,
                                duplicat: duplicate as i32,
                            },
                        )
                    } else {
                        input_used.insert(*outer_id, id);
                        (
                            *i,
                            Input {
                                typ: typ.clone(),
                                mutability: *mutability,
                                duplicat: -1,
                            },
                        )
                    }
                }
                _ => panic!("expect Operation::Input here"),
            })
            .collect::<Vec<_>>();

        let mut id_mapping = inputs
            .iter()
            .enumerate()
            .map(|(i, (aid, _))| (*aid, Id::Input(i)))
            .collect::<BTreeMap<_, _>>();

        let successors = ag.g.successors();
        let mut deg_in = ag.g.degrees_in_no_multiedge();
        let mut queue = inputs
            .iter()
            .map(|(id, _)| (*id, 0usize))
            .collect::<VecDeque<_>>();

        let mut middles = vec![vec![]];

        let mut push_to_middle = |tail: Tail, depth: usize| {
            if depth - 1 >= middles.len() {
                middles.push(vec![]);
            }
            let idx = middles[depth - 1].len();
            middles[depth - 1].push(tail);

            Id::Middle(depth - 1, idx)
        };

        while let Some((aid, depth)) = queue.pop_front() {
            let v = ag.g.vertex(aid);

            if depth != 0 {
                let tail = match &v.op {
                    Operation::Input { .. } => panic!("Operation::Input not expected here"),
                    Operation::Arith(a) => Tail::Middle(a.relabeled(&mut |aid| id_mapping[&aid])),
                    Operation::Output {
                        typ,
                        store_node,
                        in_node,
                        ..
                    } => {
                        let store_node = id_mapping[&store_node];
                        let in_node = in_node.iter().map(|in_node: &Aid| id_mapping[&in_node]).collect();
                        Tail::Output(Output {
                            typ: typ.clone(),
                            store_node,
                            in_node,
                        })
                    }
                    Operation::Todo => panic!("Operation::Todo not expected here"),
                };

                let id = push_to_middle(tail, depth);
                id_mapping.insert(aid, id);
            }

            for successor_aid in successors[aid].iter().copied() {
                deg_in[successor_aid] -= 1;
                if deg_in[successor_aid] == 0 {
                    queue.push_back((successor_aid, depth + 1));
                }
            }
        }

        NormalizedDag {
            inputs: inputs.into_iter().map(|(_, x)| x).collect(),
            middles,
        }
    }
}
