use std::collections::VecDeque;

use super::*;

#[derive(Debug)]
struct HalfMemory {
    assigned_tasks: HashSet<TaskId>,
    used: usize,
    offset: usize,
    total: usize,
    estimated_free_at: Instant,
}

impl HalfMemory {
    fn empty(offset: usize, size: usize) -> Self {
        Self {
            assigned_tasks: HashSet::new(),
            used: 0,
            offset,
            total: size,
            estimated_free_at: Instant::now(),
        }
    }

    fn free(&self) -> bool {
        self.assigned_tasks.is_empty()
    }

    fn add_task(&mut self, task: &AcceptedTask, estimated_free_at: Instant) -> usize {
        let offset = self.offset + self.used;

        let task_size = task.version.as_ref().unwrap().memory_limit() as usize;
        self.assigned_tasks.insert(task.id);
        self.used += task_size;
        self.estimated_free_at = self.estimated_free_at.max(estimated_free_at);

        offset
    }

    fn space_left(&mut self) -> usize {
        if self.assigned_tasks.is_empty() {
            self.used = 0;
            self.estimated_free_at = Instant::now();
        }
        self.total - self.used
    }
}

#[derive(Debug)]
enum AvailableMemory {
    Intact(Option<TaskId>),
    Halved([HalfMemory; 2]),
}

impl AvailableMemory {
    fn try_switch_to_intact(&mut self) {
        match &self {
            AvailableMemory::Halved([h1, h2]) if h1.free() && h2.free() => {
                *self = AvailableMemory::Intact(None)
            }
            _ => {}
        }
    }

    fn try_switch_to_halved(&mut self, total_memory: usize) {
        if let AvailableMemory::Intact(None) = &self {
            let half_size = total_memory / 2;
            *self = AvailableMemory::Halved([
                HalfMemory::empty(0, half_size),
                HalfMemory::empty(half_size, total_memory - half_size),
            ]);
        }
    }
}

#[derive(Debug)]
pub struct GlobalResources {
    available_cards: Vec<i32>,
    available_memory: AvailableMemory,
    /// Total bytes of CPU memory
    total_memory: usize,
}

impl GlobalResources {
    pub fn new(cards: Vec<i32>, available_memory: usize) -> Self {
        Self {
            available_memory: AvailableMemory::Intact(None),
            total_memory: available_memory,
            available_cards: cards,
        }
    }
}

fn try_take_cards(cards: &mut Vec<i32>, n: usize) -> Option<Vec<i32>> {
    if cards.len() >= n {
        Some((0..n).map(|_| cards.pop().unwrap()).collect())
    } else {
        None
    }
}

impl GlobalResources {
    fn try_take_cards<T>(
        &mut self,
        n: usize,
        f: impl FnOnce(Vec<i32>, &mut Self) -> Option<T>,
    ) -> Option<T> {
        let cards = try_take_cards(&mut self.available_cards, n)?;
        if let Some(r) = f(cards.clone(), self) {
            Some(r)
        } else {
            self.available_cards.extend_from_slice(&cards);
            None
        }
    }
}
pub struct Core {
    pub config: SchedulerConfig,
    /// Newly submitted tasks go here
    pending_tasks: VecDeque<AcceptedTask>,
    /// We schedule tasks window by window. by algorithms described in
    ///   Approximate Algorithms for Scheduling Parallelizable Tasks
    /// Takss from `pending_tasks` are dumped to this queue periodically
    scheduling_window: Vec<AcceptedTask>,
    scheduling_window_sorted: bool,
    scheduling_window_allotted: bool,
    resources: GlobalResources,
    pub programs: Heap<ProgramId, erased::BoxedArtifect>,
    task_id_allocator: IdAllocator<TaskId>,
    pub knowledge: Knowlede,
}

impl Core {
    pub fn new(
        config: SchedulerConfig,
        resources: GlobalResources,
        programs: Heap<ProgramId, erased::BoxedArtifect>,
    ) -> Self {
        Self {
            config: config,
            pending_tasks: VecDeque::new(),
            scheduling_window: Vec::new(),
            scheduling_window_sorted: false,
            scheduling_window_allotted: false,
            resources: resources,
            knowledge: Knowlede {
                time_experience: programs.map_by_ref(&mut |_, artifect| {
                    artifect
                        .versions()
                        .map(|ver| (ver.clone(), Duration::from_secs(1)))
                        .collect()
                }),
            },
            programs: programs,
            task_id_allocator: IdAllocator::new(),
        }
    }

    pub fn add_task(&mut self, submit: submitter::SubmittedTask) -> TaskId {
        let id = self.task_id_allocator.alloc();
        let accepted = AcceptedTask {
            id,
            submitted: submit,
            cards: Vec::new(),
            version: None,
            memory_offset: None,
        };
        self.pending_tasks.push_back(accepted);
        id
    }

    fn dump_tasks_to_scheduling_window(&mut self) {
        let n = self
            .config
            .schedule_window_size
            .min(self.pending_tasks.len());
        self.scheduling_window.extend(self.pending_tasks.drain(..n));
        self.scheduling_window_sorted = false;
        self.scheduling_window_allotted = false;
    }

    /// Determine CPU memory assigned to each task in the scheduling window.
    /// Window must be non-empty>
    fn allot_memory(&mut self) {
        // Initially set
        //   beta_j^1 = arg min_i t_j (i) dot i
        // where beta_j^k is the CPU memory allocated to task j at iteration k,
        // and t_j (i) is estimated execution of task j with i bytes assigned
        self.scheduling_window.iter_mut().for_each(|task| {
            let version = self.programs[task.submitted.program]
                .versions()
                .min_by_key(|m| {
                    self.knowledge
                        .estimate_time(task.submitted.program, *m)
                        .mul_f64(m.memory_limit_gigabytes())
                })
                .expect("artifect does not provide any version");
            task.version = Some(version.clone());
        });

        loop {
            // In each iteration k, we find task j_0 such that
            //   j0 = arg max t_j (beta_j^(k - 1))
            // and allot more memory to it
            //   beta_j0^k = arg min_(i > beta_j0^(k - 1)) t_j (i) dot i
            // while maintaining
            //   beta_j^k = beta_j^(k - 1)
            // for all j != j0
            let bottoleneck = self
                .scheduling_window
                .iter_mut()
                .max_by_key(|task| {
                    self.knowledge
                        .estimate_time(task.submitted.program, task.version.as_ref().unwrap())
                })
                .expect("empty schedule window");
            let version = self.programs[bottoleneck.submitted.program]
                .versions()
                .filter(|m| m.memory_limit() > bottoleneck.version.as_ref().unwrap().memory_limit())
                .min_by_key(|m| {
                    self.knowledge
                        .estimate_time(bottoleneck.submitted.program, *m)
                        .mul_f64(m.memory_limit_gigabytes())
                });
            if let Some(version) = version {
                bottoleneck.version = Some(version.clone());
            } else {
                self.scheduling_window_allotted = true;
                return;
            }
        }
    }

    /// Choose from schedule window a task to run.
    /// The returned [`AcceptedTask`] guarantees that the `version`, `cards` and `memory_offset` fields are set.
    pub fn schedule(&mut self) -> Option<AcceptedTask> {
        let cards_per_request = 1;

        if self.scheduling_window.is_empty() {
            self.dump_tasks_to_scheduling_window();
        }

        if self.scheduling_window.is_empty() {
            return None;
        }

        println!(
            "调度队列为 {:?}",
            self.scheduling_window
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
        );

        if !self.scheduling_window_allotted {
            self.allot_memory();
        }

        // If there are tasks consuming more than half of total, try schedule them first
        if let Some(j) = self.scheduling_window.iter().position(|task| {
            task.version.as_ref().unwrap().memory_limit() * 2 > self.resources.total_memory as u64
        }) {
            self.resources.available_memory.try_switch_to_intact();

            return self
                .resources
                .try_take_cards(cards_per_request, |cards, resources| {
                    if let AvailableMemory::Intact(None) = resources.available_memory {
                        let mut task = self.scheduling_window.remove(j);
                        task.cards = cards;
                        task.memory_offset = Some(0);
                        resources.available_memory = AvailableMemory::Intact(Some(task.id));

                        println!("调度任务 {:?} ，使用整块内存", &task);

                        Some(task)
                    } else {
                        None
                    }
                });
        }

        self.resources
            .available_memory
            .try_switch_to_halved(self.resources.total_memory);

        // Otherwise, sort the schedule window by execution time,
        // then try schedule first one to the half that is estimated to become free earlier
        if !self.scheduling_window_sorted {
            self.scheduling_window.sort_by(|a, b| {
                self.knowledge
                    .estimate_time(a.submitted.program, a.version.as_ref().unwrap())
                    .cmp(
                        &self
                            .knowledge
                            .estimate_time(b.submitted.program, b.version.as_ref().unwrap()),
                    )
                    .reverse()
            });
            self.scheduling_window_sorted = true;
        }

        if let Some(task) = self.scheduling_window.last() {
            let version = task.version.clone();

            return self
                .resources
                .try_take_cards(cards_per_request, |cards, resources| {
                    if let AvailableMemory::Halved(halves) = &mut resources.available_memory {
                        halves.sort_by_key(|halve| halve.estimated_free_at);

                        for half in halves {
                            if half.space_left()
                                >= version.as_ref().unwrap().memory_limit() as usize
                            {
                                let mut task = self.scheduling_window.pop().unwrap();
                                let estimated_free_at = Instant::now()
                                    + self.knowledge.estimate_time(
                                        task.submitted.program,
                                        task.version.as_ref().unwrap(),
                                    );
                                task.cards = cards;
                                task.memory_offset = Some(half.add_task(&task, estimated_free_at));

                                println!("调度任务 {:?} ，使用半块内存", &task);

                                return Some(task);
                            }
                        }
                    }
                    None
                });
        }

        None
    }

    pub fn finish(
        &mut self,
        id: TaskId,
        cards: &[i32],
        program: ProgramId,
        version: &MemoryInfo,
        duration: Duration,
    ) {
        self.knowledge.update(program, version, duration);

        match &mut self.resources.available_memory {
            AvailableMemory::Intact(t) => {
                if t.is_some_and(|x| x == id) {
                    self.resources.available_memory = AvailableMemory::Intact(None)
                } else {
                    panic!(
                        "finishing a task {:?} but availabel memory is Intact({:?})",
                        id, t
                    )
                }
            }
            AvailableMemory::Halved([h1, h2]) => {
                if !h1.assigned_tasks.remove(&id) && !h2.assigned_tasks.remove(&id) {
                    panic!(
                        "finishing a task {:?} but no memory is assigned to it in neither half",
                        id
                    );
                }
            }
        }

        self.resources.available_cards.extend_from_slice(cards);
    }

    pub fn resources(&self) -> &GlobalResources {
        &self.resources
    }
}
