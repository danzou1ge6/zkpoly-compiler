// This file is copied from halo2
// changes:
// - additional trait bounds
// - add wrapper for trait object
//! This module contains utilities and traits for dealing with Fiat-Shamir
//! transcripts.

use blake2b_simd::{Params as Blake2bParams, State as Blake2bState};
use group::ff::{FromUniformBytes, PrimeField};
use sha3::{Digest, Keccak256};
use std::alloc::{self, Layout};
use std::convert::TryInto;
use std::fmt::Debug;

use halo2curves::{Coordinates, CurveAffine};

use std::io::{self, Read, Write};
use std::marker::PhantomData;

use crate::args::RuntimeType;
use crate::runtime::transfer::Transfer;

/// Prefix to a prover's message soliciting a challenge
const BLAKE2B_PREFIX_CHALLENGE: u8 = 0;

/// Prefix to a prover's message containing a curve point
const BLAKE2B_PREFIX_POINT: u8 = 1;

/// Prefix to a prover's message containing a scalar
const BLAKE2B_PREFIX_SCALAR: u8 = 2;

/// Prefix to a prover's message soliciting a challenge
const KECCAK256_PREFIX_CHALLENGE: u8 = 0;

/// First prefix to a prover's message soliciting a challenge
/// Not included in the growing state!
const KECCAK256_PREFIX_CHALLENGE_LO: u8 = 10;

/// Second prefix to a prover's message soliciting a challenge
/// Not included in the growing state!
const KECCAK256_PREFIX_CHALLENGE_HI: u8 = 11;

/// Prefix to a prover's message containing a curve point
const KECCAK256_PREFIX_POINT: u8 = 1;

/// Prefix to a prover's message containing a scalar
const KECCAK256_PREFIX_SCALAR: u8 = 2;

#[derive(Clone)]
pub struct TranscriptObject<Rt: RuntimeType> {
    pub ptr: *mut Rt::Trans,
}

impl<Rt: RuntimeType> std::fmt::Debug for TranscriptObject<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { self.ptr.as_ref().fmt(f) }
    }
}

impl<Rt: RuntimeType> TranscriptObject<Rt> {
    pub fn new(val: Rt::Trans) -> Self {
        unsafe {
            let ptr = alloc::alloc(Layout::new::<Rt::Trans>()) as *mut Rt::Trans;
            ptr.write(val);
            Self { ptr }
        }
    }

    pub fn new_raw() -> Self {
        unsafe {
            let ptr = alloc::alloc(Layout::new::<Rt::Trans>()) as *mut Rt::Trans;
            Self { ptr }
        }
    }

    pub fn deallocate(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr as *mut u8, Layout::new::<Rt::Trans>());
        }
    }

    pub fn as_ref(&self) -> &Rt::Trans {
        unsafe { &*self.ptr }
    }

    pub fn as_mut(&mut self) -> &mut Rt::Trans {
        unsafe { &mut *self.ptr }
    }

    pub fn take(self) -> Rt::Trans {
        unsafe {
            let r = std::ptr::read(self.ptr);
            alloc::dealloc(self.ptr as *mut u8, Layout::new::<Rt::Trans>());
            r
        }
    }
}

impl<Rt: RuntimeType> Transfer for TranscriptObject<Rt> {
    fn cpu2cpu(&self, other: &mut Self) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, other.ptr, 1);
        }
    }
}

unsafe impl<Rt: RuntimeType> Send for TranscriptObject<Rt> {}
unsafe impl<Rt: RuntimeType> Sync for TranscriptObject<Rt> {}
/// Generic transcript view (from either the prover or verifier's perspective)
pub trait Transcript<C: CurveAffine, E: EncodedChallenge<C>>: Send + Sync + Clone {
    /// Squeeze an encoded verifier challenge from the transcript.
    fn squeeze_challenge(&mut self) -> E;

    /// Squeeze a typed challenge (in the scalar field) from the transcript.
    fn squeeze_challenge_scalar<T>(&mut self) -> ChallengeScalar<C, T> {
        ChallengeScalar {
            inner: self.squeeze_challenge().get_scalar(),
            _marker: PhantomData,
        }
    }

    /// Writing the point to the transcript without writing it to the proof,
    /// treating it as a common input.
    fn common_point(&mut self, point: C) -> io::Result<()>;

    /// Writing the scalar to the transcript without writing it to the proof,
    /// treating it as a common input.
    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()>;
}

/// Transcript view from the perspective of a verifier that has access to an
/// input stream of data from the prover to the verifier.
pub trait TranscriptRead<C: CurveAffine, E: EncodedChallenge<C>>: Transcript<C, E> {
    /// Read a curve point from the prover.
    fn read_point(&mut self) -> io::Result<C>;

    /// Read a curve scalar from the prover.
    fn read_scalar(&mut self) -> io::Result<C::Scalar>;
}

/// Transcript view from the perspective of a prover that has access to an
/// output stream of messages from the prover to the verifier.
pub trait TranscriptWrite<C: CurveAffine, E: EncodedChallenge<C>>:
    Transcript<C, E> + Debug
{
    /// Write a curve point to the proof and the transcript.
    fn write_point(&mut self, point: C) -> io::Result<()>;

    /// Write a scalar to the proof and the transcript.
    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()>;
}

/// Initializes transcript at verifier side.
pub trait TranscriptReadBuffer<R: Read, C: CurveAffine, E: EncodedChallenge<C>>:
    TranscriptRead<C, E>
{
    /// Initialize a transcript given an input buffer.
    fn init(reader: R) -> Self;
}

/// Manages beginning and finishing of transcript pipeline.
pub trait TranscriptWriterBuffer<W: Write, C: CurveAffine, E: EncodedChallenge<C>>:
    TranscriptWrite<C, E>
{
    /// Initialize a transcript given an output buffer.
    fn init(writer: W) -> Self;

    /// Conclude the interaction and return the output buffer (writer).
    fn finalize(self) -> W;
}

/// We will replace BLAKE2b with an algebraic hash function in a later version.
#[derive(Debug, Clone)]
pub struct Blake2bRead<R: Read + Send + Sync, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Blake2bState,
    reader: R,
    _marker: PhantomData<(C, E)>,
}

/// Keccak256 hash function reader for EVM compatibility
#[derive(Debug, Clone)]
pub struct Keccak256Read<R: Read + Send + Sync, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Keccak256,
    reader: R,
    _marker: PhantomData<(C, E)>,
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> TranscriptReadBuffer<R, C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Initialize a transcript given an input buffer.
    fn init(reader: R) -> Self {
        Blake2bRead {
            state: Blake2bParams::new()
                .hash_length(64)
                .personal(b"Halo2-Transcript")
                .to_state(),
            reader,
            _marker: PhantomData,
        }
    }
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> TranscriptReadBuffer<R, C, Challenge255<C>>
    for Keccak256Read<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Initialize a transcript given an input buffer.
    fn init(reader: R) -> Self {
        let mut state = Keccak256::new();
        state.update(b"Halo2-Transcript");
        Keccak256Read {
            state,
            reader,
            _marker: PhantomData,
        }
    }
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> TranscriptRead<C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn read_point(&mut self) -> io::Result<C> {
        let mut compressed = C::Repr::default();
        self.reader.read_exact(compressed.as_mut())?;
        let point: C = Option::from(C::from_bytes(&compressed)).ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof")
        })?;
        self.common_point(point)?;

        Ok(point)
    }

    fn read_scalar(&mut self) -> io::Result<C::Scalar> {
        let mut data = <C::Scalar as PrimeField>::Repr::default();
        self.reader.read_exact(data.as_mut())?;
        let scalar: C::Scalar = Option::from(C::Scalar::from_repr(data)).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid field element encoding in proof",
            )
        })?;
        self.common_scalar(scalar)?;

        Ok(scalar)
    }
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> TranscriptRead<C, Challenge255<C>>
    for Keccak256Read<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn read_point(&mut self) -> io::Result<C> {
        let mut compressed = C::Repr::default();
        self.reader.read_exact(compressed.as_mut())?;
        let point: C = Option::from(C::from_bytes(&compressed)).ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof")
        })?;
        self.common_point(point)?;

        Ok(point)
    }

    fn read_scalar(&mut self) -> io::Result<C::Scalar> {
        let mut data = <C::Scalar as PrimeField>::Repr::default();
        self.reader.read_exact(data.as_mut())?;
        let scalar: C::Scalar = Option::from(C::Scalar::from_repr(data)).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid field element encoding in proof",
            )
        })?;
        self.common_scalar(scalar)?;

        Ok(scalar)
    }
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update(&[BLAKE2B_PREFIX_CHALLENGE]);
        let hasher = self.state.clone();
        let result: [u8; 64] = hasher.finalize().as_bytes().try_into().unwrap();
        Challenge255::<C>::new(&result)
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        self.state.update(coords.x().to_repr().as_ref());
        self.state.update(coords.y().to_repr().as_ref());

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}

impl<R: Read + Sync + Send + Clone, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Keccak256Read<R, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update([KECCAK256_PREFIX_CHALLENGE]);

        let mut state_lo = self.state.clone();
        let mut state_hi = self.state.clone();
        state_lo.update([KECCAK256_PREFIX_CHALLENGE_LO]);
        state_hi.update([KECCAK256_PREFIX_CHALLENGE_HI]);
        let result_lo: [u8; 32] = state_lo.finalize().as_slice().try_into().unwrap();
        let result_hi: [u8; 32] = state_hi.finalize().as_slice().try_into().unwrap();

        let mut t = result_lo.to_vec();
        t.extend_from_slice(&result_hi[..]);
        let result: [u8; 64] = t.as_slice().try_into().unwrap();

        Challenge255::<C>::new(&result)
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update([KECCAK256_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        self.state.update(coords.x().to_repr().as_ref());
        self.state.update(coords.y().to_repr().as_ref());

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update([KECCAK256_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}

/// We will replace BLAKE2b with an algebraic hash function in a later version.
#[derive(Debug, Clone)]
pub struct Blake2bWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Blake2bState,
    writer: W,
    _marker: PhantomData<(C, E)>,
}

/// Keccak256 hash function writer for EVM compatibility
#[derive(Debug, Clone)]
pub struct Keccak256Write<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Keccak256,
    writer: W,
    _marker: PhantomData<(C, E)>,
}

impl<W: Write + Sync + Send + Clone + Debug, C: CurveAffine>
    TranscriptWriterBuffer<W, C, Challenge255<C>> for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Initialize a transcript given an output buffer.
    fn init(writer: W) -> Self {
        Blake2bWrite {
            state: Blake2bParams::new()
                .hash_length(64)
                .personal(b"Halo2-Transcript")
                .to_state(),
            writer,
            _marker: PhantomData,
        }
    }

    fn finalize(self) -> W {
        // TODO: handle outstanding scalars? see issue #138
        self.writer
    }
}

impl<W: Write + Sync + Send + Clone + Debug, C: CurveAffine>
    TranscriptWriterBuffer<W, C, Challenge255<C>> for Keccak256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Initialize a transcript given an output buffer.
    fn init(writer: W) -> Self {
        let mut state = Keccak256::new();
        state.update(b"Halo2-Transcript");
        Keccak256Write {
            state,
            writer,
            _marker: PhantomData,
        }
    }

    /// Conclude the interaction and return the output buffer (writer).
    fn finalize(self) -> W {
        // TODO: handle outstanding scalars? see issue #138
        self.writer
    }
}

impl<W: Write + Sync + Send + Clone + Debug, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        let compressed = point.to_bytes();
        self.writer.write_all(compressed.as_ref())
    }
    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();
        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write + Sync + Send + Clone + Debug, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for Keccak256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        let compressed = point.to_bytes();
        self.writer.write_all(compressed.as_ref())
    }
    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();
        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write + Sync + Send + Clone, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update(&[BLAKE2B_PREFIX_CHALLENGE]);
        let hasher = self.state.clone();
        let result: [u8; 64] = hasher.finalize().as_bytes().try_into().unwrap();
        Challenge255::<C>::new(&result)
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        self.state.update(coords.x().to_repr().as_ref());
        self.state.update(coords.y().to_repr().as_ref());

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}

impl<W: Write + Sync + Send + Clone, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Keccak256Write<W, C, Challenge255<C>>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update([KECCAK256_PREFIX_CHALLENGE]);

        let mut state_lo = self.state.clone();
        let mut state_hi = self.state.clone();
        state_lo.update([KECCAK256_PREFIX_CHALLENGE_LO]);
        state_hi.update([KECCAK256_PREFIX_CHALLENGE_HI]);
        let result_lo: [u8; 32] = state_lo.finalize().as_slice().try_into().unwrap();
        let result_hi: [u8; 32] = state_hi.finalize().as_slice().try_into().unwrap();

        let mut t = result_lo.to_vec();
        t.extend_from_slice(&result_hi[..]);
        let result: [u8; 64] = t.as_slice().try_into().unwrap();

        Challenge255::<C>::new(&result)
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update([KECCAK256_PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        self.state.update(coords.x().to_repr().as_ref());
        self.state.update(coords.y().to_repr().as_ref());

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update([KECCAK256_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}

/// The scalar representation of a verifier challenge.
///
/// The `Type` type can be used to scope the challenge to a specific context, or
/// set to `()` if no context is required.
#[derive(Copy, Clone, Debug)]
pub struct ChallengeScalar<C: CurveAffine, T> {
    inner: C::Scalar,
    _marker: PhantomData<T>,
}

pub type ChallengeScalarUnit<C> = ChallengeScalar<C, ()>;

impl<C: CurveAffine, T> std::ops::Deref for ChallengeScalar<C, T> {
    type Target = C::Scalar;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// `EncodedChallenge<C>` defines a challenge encoding with a [`Self::Input`]
/// that is used to derive the challenge encoding and `get_challenge` obtains
/// the _real_ `C::Scalar` that the challenge encoding represents.
pub trait EncodedChallenge<C: CurveAffine> {
    /// The Input type used to derive the challenge encoding. For example,
    /// an input from the Poseidon hash would be a base field element;
    /// an input from the Blake2b hash would be a [u8; 64].
    type Input;

    /// Get an encoded challenge from a given input challenge.
    fn new(challenge_input: &Self::Input) -> Self;

    /// Get a scalar field element from an encoded challenge.
    fn get_scalar(&self) -> C::Scalar;

    /// Cast an encoded challenge as a typed `ChallengeScalar`.
    fn as_challenge_scalar<T>(&self) -> ChallengeScalar<C, T> {
        ChallengeScalar {
            inner: self.get_scalar(),
            _marker: PhantomData,
        }
    }
}

/// A 255-bit challenge.
#[derive(Copy, Clone, Debug)]
pub struct Challenge255<C: CurveAffine>([u8; 32], PhantomData<C>);

impl<C: CurveAffine> std::ops::Deref for Challenge255<C> {
    type Target = [u8; 32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: CurveAffine> EncodedChallenge<C> for Challenge255<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    type Input = [u8; 64];

    fn new(challenge_input: &[u8; 64]) -> Self {
        Challenge255(
            C::Scalar::from_uniform_bytes(challenge_input)
                .to_repr()
                .as_ref()
                .try_into()
                .expect("Scalar fits into 256 bits"),
            PhantomData,
        )
    }
    fn get_scalar(&self) -> C::Scalar {
        let mut repr = <C::Scalar as PrimeField>::Repr::default();
        repr.as_mut().copy_from_slice(&self.0);
        C::Scalar::from_repr(repr).unwrap()
    }
}

pub fn read_n_points<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    n: usize,
) -> io::Result<Vec<C>> {
    (0..n).map(|_| transcript.read_point()).collect()
}

pub fn read_n_scalars<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    n: usize,
) -> io::Result<Vec<C::Scalar>> {
    (0..n).map(|_| transcript.read_scalar()).collect()
}
