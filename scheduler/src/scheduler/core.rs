use super::*;

pub struct Core {
    pub config: SchedulerConfig,
    /// Newly submitted tasks go here
    pending_tasks: Vec<AcceptedTask>,
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
            pending_tasks: Vec::new(),
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
        self.pending_tasks.push(accepted);
        id
    }

    fn dump_tasks_to_scheduling_window(&mut self) {
        self.scheduling_window
            .extend(std::mem::take(&mut self.pending_tasks).into_iter());
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

            if let Some(cards) = self.resources.try_take_cards(self.config.cards_per_request) {
                if let AvailableMemory::Intact(None) = self.resources.available_memory {
                    let mut task = self.scheduling_window.remove(j);
                    task.cards = cards;
                    task.memory_offset = Some(0);
                    self.resources.available_memory = AvailableMemory::Intact(Some(task.id));

                    println!("调度任务 {:?} ，使用整块内存", &task);

                    return Some(task);
                }
            }
            return None;
        }

        self.resources
            .available_memory
            .try_switch_to_halved(self.resources.total_memory);

        // Otherwise, sort the schedule window by execution time,
        // then try schedule first one to the half that is estimated to become free earlier
        if !self.scheduling_window_sorted {
            self.scheduling_window.sort_by(|a, b| {
                self.knowledge
                    .estimate_time(a.submitted.program, b.version.as_ref().unwrap())
                    .cmp(
                        &self
                            .knowledge
                            .estimate_time(a.submitted.program, b.version.as_ref().unwrap()),
                    )
                    .reverse()
            });
            self.scheduling_window_sorted = true;
        }

        if let Some(task) = self.scheduling_window.last() {
            if let AvailableMemory::Halved(halves) = &mut self.resources.available_memory {
                halves.sort_by_key(|halve| halve.estimated_free_at);

                for half in halves {
                    if let Some(cards) = try_take_cards(
                        &mut self.resources.available_cards,
                        self.config.cards_per_request,
                    ) {
                        if half.space_left()
                            >= task.version.as_ref().unwrap().memory_limit() as usize
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
            }
            return None;
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
}
