use std::marker::PhantomData;

use super::erased;
use super::prelude::*;

define_usize_id!(ProgramId);

pub mod exposed {
    use super::*;

    pub struct RunReturn<Rt: RuntimeType> {
        pub ret_value: Option<Variable<Rt>>,
        pub log: Log,
        pub time: Duration,
    }
    pub struct SubmittedTask<Rt: RuntimeType> {
        pub(super) program: ProgramToken<Rt>,
        pub(super) inputs: EntryTable<Rt>,
        pub(super) debug_opt: RuntimeDebug,
    }
    impl<Rt: RuntimeType> SubmittedTask<Rt> {
        pub fn new(program: ProgramToken<Rt>, inputs: EntryTable<Rt>) -> Self {
            Self {
                program,
                inputs,
                debug_opt: RuntimeDebug::none(),
            }
        }

        pub fn with_debug_opt(self, x: RuntimeDebug) -> Self {
            Self {
                debug_opt: x,
                ..self
            }
        }
    }

    #[derive(Debug, Clone)]
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

    pub struct Programs<Rt: RuntimeType>(pub(super) Heap<ProgramId, Artifect<Rt>>);

    impl<Rt: RuntimeType> Programs<Rt> {
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
    pub(super) debug_opt: RuntimeDebug,
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
            debug_opt: value.debug_opt,
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

#[derive(Debug, Clone)]
pub struct Submitter<Rt: RuntimeType> {
    pub(super) sender: mpsc::Sender<Message>,
    pub(super) _phantom: PhantomData<Rt>,
}

pub type SubmitResult<T> = Result<T, mpsc::SendError<Message>>;

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

    pub fn add_artifect(&self, artifect: Artifect<Rt>) -> SubmitResult<exposed::ProgramToken<Rt>> {
        let (sender, receiver) = mpsc::channel::<ProgramId>();

        self.sender.send(Message::Add(Box::new(artifect), sender))?;
        let id = receiver.recv().unwrap();

        Ok(exposed::ProgramToken::new(id))
    }

    pub fn alternative_rt<Rt2: RuntimeType>(&self) -> Submitter<Rt2> {
        Submitter {
            sender: self.sender.clone(),
            _phantom: PhantomData,
        }
    }
}
