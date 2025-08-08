use std::marker::PhantomData;

use super::erased;
use super::prelude::*;

define_usize_id!(ProgramId);

pub mod exposed {
    use super::*;

    /// The result of task exeuction.
    pub struct RunReturn<Rt: RuntimeType> {
        /// Return value of the computation graph.
        pub ret_value: Option<Variable<Rt>>,
        pub log: Log,
        /// Total time consumed to run the computation.
        pub time: Duration,
    }

    /// The taks to submit to the scheduler.
    pub struct SubmittedTask<Rt: RuntimeType> {
        pub(super) program: ProgramToken<Rt>,
        pub(super) inputs: EntryTable<Rt>,
    }

    impl<Rt: RuntimeType> SubmittedTask<Rt> {
        /// Create a task that invokes artifect `program` with `inputs`.
        pub fn new(program: ProgramToken<Rt>, inputs: EntryTable<Rt>) -> Self {
            Self { program, inputs }
        }
    }

    #[derive(Debug, Clone)]
    /// Unique identifier for each artifect added to the scheduler.
    pub struct ProgramToken<Rt: RuntimeType> {
        pub(super) id: ProgramId,
        _phantom: PhantomData<Rt>,
    }

    impl<Rt: RuntimeType> Copy for ProgramToken<Rt> {}

    impl<Rt: RuntimeType> ProgramToken<Rt> {
        pub(super) fn new(id: ProgramId) -> Self {
            Self {
                id,
                _phantom: PhantomData,
            }
        }
    }

    /// A collection of artifects to put in the scheduler.
    pub struct Programs<Rt: RuntimeType>(pub(super) Heap<ProgramId, Artifect<Rt>>);

    impl<Rt: RuntimeType> Programs<Rt> {
        /// Add artifect to the collection, returnning its identifier.
        pub fn push(&mut self, artifect: Artifect<Rt>) -> ProgramToken<Rt> {
            let id = self.0.push(artifect);
            ProgramToken::new(id)
        }

        pub fn new() -> Self {
            Self(Heap::new())
        }
    }
}

impl<Rt: RuntimeType> exposed::Programs<Rt> {
    pub(super) fn erase(self) -> Heap<ProgramId, erased::BoxedArtifect> {
        self.0
            .map(&mut |_, ar| Box::new(ar) as erased::BoxedArtifect)
    }
}

pub(super) struct SubmittedTask {
    pub(super) program: ProgramId,
    pub(super) inputs: erased::BoxedVariableTable,
}

pub(super) struct RunReturn {
    pub(super) ret_value: Option<erased::BoxedVariable>,
    pub(super) log: Log,
    pub(super) time: Duration,
}

impl<Rt: RuntimeType> From<exposed::SubmittedTask<Rt>> for SubmittedTask {
    fn from(value: exposed::SubmittedTask<Rt>) -> Self {
        Self {
            program: value.program.id,
            inputs: Box::new(value.inputs),
        }
    }
}

impl<Rt: RuntimeType> From<RunReturn> for exposed::RunReturn<Rt> {
    fn from(value: RunReturn) -> Self {
        Self {
            ret_value: value.ret_value.map(|x| erased::downcast_variable(x)),
            log: value.log,
            time: value.time,
        }
    }
}

pub struct Submit {
    pub(super) task: SubmittedTask,
    pub(super) result_sender: crossbeam_channel::Sender<RunReturn>,
}

pub enum Message {
    Submit(Submit),
    Add(erased::BoxedArtifect, mpsc::Sender<ProgramId>),
}

/// A submitter connected to some scheduler.
#[derive(Debug, Clone)]
pub struct Submitter<Rt: RuntimeType> {
    pub(super) sender: mpsc::Sender<Message>,
    pub(super) _phantom: PhantomData<Rt>,
}

pub type SubmitResult<T> = Result<T, mpsc::SendError<Message>>;

/// The result of the submitted task, yet to be fulfilled.
/// Internally, [`Future`] contains a channel receiver where the scheduler
/// will put result after task has completed.
pub struct Future<Rt: RuntimeType> {
    receiver: crossbeam_channel::Receiver<RunReturn>,
    _phantom: PhantomData<Rt>,
}

impl<Rt: RuntimeType> Future<Rt> {
    /// Block current thread until future is fulfilled, or the scheduler has exited.
    /// Upon wake up, either returns the result or errors if scheduler exited.
    pub fn read(&self) -> Result<exposed::RunReturn<Rt>, crossbeam_channel::RecvError> {
        let r = self.receiver.recv()?;
        Ok(r.into())
    }

    /// Return result if future is fulfilled or errors if future is pending or scheduler exited.
    pub fn try_read(&self) -> Result<exposed::RunReturn<Rt>, crossbeam_channel::TryRecvError> {
        let r = self.receiver.try_recv()?;
        Ok(r.into())
    }

    /// Whether the future is fulfilled, that is, contains valid value
    pub fn ready(&self) -> bool {
        !self.receiver.is_empty()
    }
}

impl<Rt: RuntimeType> Submitter<Rt> {
    /// Submit a task to the scheduler.
    pub fn submit(&self, task: exposed::SubmittedTask<Rt>) -> SubmitResult<Future<Rt>> {
        let (sender, receiver) = crossbeam_channel::unbounded::<RunReturn>();

        self.sender.send(Message::Submit(Submit {
            task: task.into(),
            result_sender: sender,
        }))?;

        Ok(Future {
            receiver,
            _phantom: PhantomData,
        })
    }

    /// Add a new artifect to the scheduler.
    pub fn add_artifect(&self, artifect: Artifect<Rt>) -> SubmitResult<exposed::ProgramToken<Rt>> {
        let (sender, receiver) = mpsc::channel::<ProgramId>();

        self.sender.send(Message::Add(Box::new(artifect), sender))?;
        let id = receiver.recv().unwrap();

        Ok(exposed::ProgramToken::new(id))
    }

    /// Clone to a submitter that runs on different [`RuntimeType`], but still connected to the
    /// same scheduler.
    pub fn alternative_rt<Rt2: RuntimeType>(&self) -> Submitter<Rt2> {
        Submitter {
            sender: self.sender.clone(),
            _phantom: PhantomData,
        }
    }
}
