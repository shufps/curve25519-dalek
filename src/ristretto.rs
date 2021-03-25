// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2019 Isis Lovecruft, Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - Isis Agora Lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>

// We allow non snake_case names because coordinates in projective space are
// traditionally denoted by the capitalisation of their respective
// counterparts in affine space.  Yeah, you heard me, rustc, I'm gonna have my
// affine and projective cakes and eat both of them too.
#![allow(non_snake_case)]

//! An implementation of [Ristretto][ristretto_main], which provides a
//! prime-order group.
//!
//! # The Ristretto Group
//!
//! Ristretto is a modification of Mike Hamburg's Decaf scheme to work
//! with cofactor-\\(8\\) curves, such as Curve25519.
//!
//! The introduction of the Decaf paper, [_Decaf:
//! Eliminating cofactors through point
//! compression_](https://eprint.iacr.org/2015/673.pdf), notes that while
//! most cryptographic systems require a group of prime order, most
//! concrete implementations using elliptic curve groups fall short –
//! they either provide a group of prime order, but with incomplete or
//! variable-time addition formulae (for instance, most Weierstrass
//! models), or else they provide a fast and safe implementation of a
//! group whose order is not quite a prime \\(q\\), but \\(hq\\) for a
//! small cofactor \\(h\\) (for instance, Edwards curves, which have
//! cofactor at least \\(4\\)).
//!
//! This abstraction mismatch is commonly “handled” by pushing the
//! complexity upwards, adding ad-hoc protocol modifications.  But
//! these modifications require careful analysis and are a recurring
//! source of [vulnerabilities][cryptonote] and [design
//! complications][ed25519_hkd].
//!
//! Instead, Decaf (and Ristretto) use a quotient group to implement a
//! prime-order group using a non-prime-order curve.  This provides
//! the correct abstraction for cryptographic systems, while retaining
//! the speed and safety benefits of an Edwards curve.
//!
//! Decaf is named “after the procedure which divides the effect of
//! coffee by \\(4\\)”.  However, Curve25519 has a cofactor of
//! \\(8\\).  To eliminate its cofactor, Ristretto restricts further;
//! this [additional restriction][ristretto_coffee] gives the
//! _Ristretto_ encoding.
//!
//! More details on why Ristretto is necessary can be found in the
//! [Why Ristretto?][why_ristretto] section of the Ristretto website.
//!
//! Ristretto
//! points are provided in `curve25519-dalek` by the `RistrettoPoint`
//! struct.
//!
//! ## Encoding and Decoding
//!
//! Encoding is done by converting to and from a `CompressedRistretto`
//! struct, which is a typed wrapper around `[u8; 32]`.
//!
//! The encoding is not batchable, but it is possible to
//! double-and-encode in a batch using
//! `RistrettoPoint::double_and_compress_batch`.
//!
//! ## Equality Testing
//!
//! Testing equality of points on an Edwards curve in projective
//! coordinates requires an expensive inversion.  By contrast, equality
//! checking in the Ristretto group can be done in projective
//! coordinates without requiring an inversion, so it is much faster.
//!
//! The `RistrettoPoint` struct implements the
//! `subtle::ConstantTimeEq` trait for constant-time equality
//! checking, and the Rust `Eq` trait for variable-time equality
//! checking.
//!
//! ## Scalars
//!
//! Scalars are represented by the `Scalar` struct.  Each scalar has a
//! canonical representative mod the group order.  To attempt to load
//! a supposedly-canonical scalar, use
//! `Scalar::from_canonical_bytes()`. To check whether a
//! representative is canonical, use `Scalar::is_canonical()`.
//!
//! ## Scalar Multiplication
//!
//! Scalar multiplication on Ristretto points is provided by:
//!
//! * the `*` operator between a `Scalar` and a `RistrettoPoint`, which
//! performs constant-time variable-base scalar multiplication;
//!
//! * the `*` operator between a `Scalar` and a
//! `RistrettoBasepointTable`, which performs constant-time fixed-base
//! scalar multiplication;
//!
//! * an implementation of the
//! [`MultiscalarMul`](../traits/trait.MultiscalarMul.html) trait for
//! constant-time variable-base multiscalar multiplication;
//!
//! * an implementation of the
//! [`VartimeMultiscalarMul`](../traits/trait.VartimeMultiscalarMul.html)
//! trait for variable-time variable-base multiscalar multiplication;
//!
//! ## Random Points and Hashing to Ristretto
//!
//! The Ristretto group comes equipped with an Elligator map.  This is
//! used to implement
//!
//! * `RistrettoPoint::random()`, which generates random points from an
//! RNG;
//!
//! * `RistrettoPoint::from_hash()` and
//! `RistrettoPoint::hash_from_bytes()`, which perform hashing to the
//! group.
//!
//! The Elligator map itself is not currently exposed.
//!
//! ## Implementation
//!
//! The Decaf suggestion is to use a quotient group, such as \\(\mathcal
//! E / \mathcal E[4]\\) or \\(2 \mathcal E / \mathcal E[2] \\), to
//! implement a prime-order group using a non-prime-order curve.
//!
//! This requires only changing
//!
//! 1. the function for equality checking (so that two representatives
//!    of the same coset are considered equal);
//! 2. the function for encoding (so that two representatives of the
//!    same coset are encoded as identical bitstrings);
//! 3. the function for decoding (so that only the canonical encoding of
//!    a coset is accepted).
//!
//! Internally, each coset is represented by a curve point; two points
//! \\( P, Q \\) may represent the same coset in the same way that two
//! points with different \\(X,Y,Z\\) coordinates may represent the
//! same point.  The group operations are carried out with no overhead
//! using Edwards formulas.
//!
//! Notes on the details of the encoding can be found in the
//! [Details][ristretto_notes] section of the Ristretto website.
//!
//! [cryptonote]:
//! https://moderncrypto.org/mail-archive/curves/2017/000898.html
//! [ed25519_hkd]:
//! https://moderncrypto.org/mail-archive/curves/2017/000858.html
//! [ristretto_coffee]:
//! https://en.wikipedia.org/wiki/Ristretto
//! [ristretto_notes]:
//! https://ristretto.group/details/index.html
//! [why_ristretto]:
//! https://ristretto.group/why_ristretto.html
//! [ristretto_main]:
//! https://ristretto.group/

use core::borrow::Borrow;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Neg, Sub};
use core::ops::{AddAssign, SubAssign};
use core::ops::{Mul, MulAssign};


use digest::generic_array::typenum::U64;
use digest::Digest;

use constants;
use field::FieldElement;

use subtle::Choice;
use subtle::ConditionallySelectable;
use subtle::ConditionallyNegatable;
use subtle::ConstantTimeEq;

use edwards::EdwardsBasepointTable;
use edwards::EdwardsPoint;

#[allow(unused_imports)]
use prelude::*;

use scalar::Scalar;

use traits::Identity;
#[cfg(any(feature = "alloc", feature = "std"))]
use traits::{MultiscalarMul, VartimeMultiscalarMul, VartimePrecomputedMultiscalarMul};

#[cfg(not(all(
    feature = "simd_backend",
    any(target_feature = "avx2", target_feature = "avx512ifma")
)))]
use backend::serial::scalar_mul;
#[cfg(all(
    feature = "simd_backend",
    any(target_feature = "avx2", target_feature = "avx512ifma")
))]
use backend::vector::scalar_mul;

// ------------------------------------------------------------------------
// Compressed points
// ------------------------------------------------------------------------

/// A Ristretto point, in compressed wire format.
///
/// The Ristretto encoding is canonical, so two points are equal if and
/// only if their encodings are equal.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CompressedRistretto(pub [u8; 32]);

impl ConstantTimeEq for CompressedRistretto {
    fn ct_eq(&self, other: &CompressedRistretto) -> Choice {
        self.as_bytes().ct_eq(other.as_bytes())
    }
}

impl CompressedRistretto {
    /// Copy the bytes of this `CompressedRistretto`.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// View this `CompressedRistretto` as an array of bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Construct a `CompressedRistretto` from a slice of bytes.
    ///
    /// # Panics
    ///
    /// If the input `bytes` slice does not have a length of 32.
    pub fn from_slice(bytes: &[u8]) -> CompressedRistretto {
        let mut tmp = [0u8; 32];

        tmp.copy_from_slice(bytes);

        CompressedRistretto(tmp)
    }

    /// Attempt to decompress to an `RistrettoPoint`.
    ///
    /// # Return
    ///
    /// - `Some(RistrettoPoint)` if `self` was the canonical encoding of a point;
    ///
    /// - `None` if `self` was not the canonical encoding of a point.
    pub fn decompress(&self) -> Option<RistrettoPoint> {
        // Step 1. Check s for validity:
        // 1.a) s must be 32 bytes (we get this from the type system)
        // 1.b) s < p
        // 1.c) s is nonnegative
        //
        // Our decoding routine ignores the high bit, so the only
        // possible failure for 1.b) is if someone encodes s in 0..18
        // as s+p in 2^255-19..2^255-1.  We can check this by
        // converting back to bytes, and checking that we get the
        // original input, since our encoding routine is canonical.

        let s = FieldElement::from_bytes(self.as_bytes());
        let s_bytes_check = s.to_bytes();
        let s_encoding_is_canonical =
            &s_bytes_check[..].ct_eq(self.as_bytes());
        let s_is_negative = s.is_negative();

        if s_encoding_is_canonical.unwrap_u8() == 0u8 || s_is_negative.unwrap_u8() == 1u8 {
            return None;
        }

        // Step 2.  Compute (X:Y:Z:T).
        let one = FieldElement::one();
        let ss = s.square();
        let u1 = &one - &ss;      //  1 + as²
        let u2 = &one + &ss;      //  1 - as²    where a=-1
        let u2_sqr = u2.square(); // (1 - as²)²

        // v == ad(1+as²)² - (1-as²)²            where d=-121665/121666
        let v = &(&(-&constants::EDWARDS_D) * &u1.square()) - &u2_sqr;

        let (ok, I) = (&v * &u2_sqr).invsqrt(); // 1/sqrt(v*u_2²)

        let Dx = &I * &u2;         // 1/sqrt(v)
        let Dy = &I * &(&Dx * &v); // 1/u2

        // x == | 2s/sqrt(v) | == + sqrt(4s²/(ad(1+as²)² - (1-as²)²))
        let mut x = &(&s + &s) * &Dx;
        let x_neg = x.is_negative();
        x.conditional_negate(x_neg);

        // y == (1-as²)/(1+as²)
        let y = &u1 * &Dy;

        // t == ((1+as²) sqrt(4s²/(ad(1+as²)² - (1-as²)²)))/(1-as²)
        let t = &x * &y;

        if ok.unwrap_u8() == 0u8 || t.is_negative().unwrap_u8() == 1u8 || y.is_zero().unwrap_u8() == 1u8 {
            None
        } else {
            Some(RistrettoPoint(EdwardsPoint{X: x, Y: y, Z: one, T: t}))
        }
    }
}

impl Identity for CompressedRistretto {
    fn identity() -> CompressedRistretto {
        CompressedRistretto([0u8; 32])
    }
}

impl Default for CompressedRistretto {
    fn default() -> CompressedRistretto {
        CompressedRistretto::identity()
    }
}

// ------------------------------------------------------------------------
// Serde support
// ------------------------------------------------------------------------
// Serializes to and from `RistrettoPoint` directly, doing compression
// and decompression internally.  This means that users can create
// structs containing `RistrettoPoint`s and use Serde's derived
// serializers to serialize those structures.

#[cfg(feature = "serde")]
use serde::{self, Serialize, Deserialize, Serializer, Deserializer};
#[cfg(feature = "serde")]
use serde::de::Visitor;

#[cfg(feature = "serde")]
impl Serialize for RistrettoPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(32)?;
        for byte in self.compress().as_bytes().iter() {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

#[cfg(feature = "serde")]
impl Serialize for CompressedRistretto {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(32)?;
        for byte in self.as_bytes().iter() {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RistrettoPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        struct RistrettoPointVisitor;

        impl<'de> Visitor<'de> for RistrettoPointVisitor {
            type Value = RistrettoPoint;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid point in Ristretto format")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<RistrettoPoint, A::Error>
                where A: serde::de::SeqAccess<'de>
            {
                let mut bytes = [0u8; 32];
                for i in 0..32 {
                    bytes[i] = seq.next_element()?
                        .ok_or(serde::de::Error::invalid_length(i, &"expected 32 bytes"))?;
                }
                CompressedRistretto(bytes)
                    .decompress()
                    .ok_or(serde::de::Error::custom("decompression failed"))
            }
        }

        deserializer.deserialize_tuple(32, RistrettoPointVisitor)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for CompressedRistretto {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        struct CompressedRistrettoVisitor;

        impl<'de> Visitor<'de> for CompressedRistrettoVisitor {
            type Value = CompressedRistretto;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("32 bytes of data")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<CompressedRistretto, A::Error>
                where A: serde::de::SeqAccess<'de>
            {
                let mut bytes = [0u8; 32];
                for i in 0..32 {
                    bytes[i] = seq.next_element()?
                        .ok_or(serde::de::Error::invalid_length(i, &"expected 32 bytes"))?;
                }
                Ok(CompressedRistretto(bytes))
            }
        }

        deserializer.deserialize_tuple(32, CompressedRistrettoVisitor)
    }
}

// ------------------------------------------------------------------------
// Internal point representations
// ------------------------------------------------------------------------

/// A `RistrettoPoint` represents a point in the Ristretto group for
/// Curve25519.  Ristretto, a variant of Decaf, constructs a
/// prime-order group as a quotient group of a subgroup of (the
/// Edwards form of) Curve25519.
///
/// Internally, a `RistrettoPoint` is implemented as a wrapper type
/// around `EdwardsPoint`, with custom equality, compression, and
/// decompression routines to account for the quotient.  This means that
/// operations on `RistrettoPoint`s are exactly as fast as operations on
/// `EdwardsPoint`s.
///
#[derive(Copy, Clone)]
pub struct RistrettoPoint(pub(crate) EdwardsPoint);

impl RistrettoPoint {
    /// Compress this point using the Ristretto encoding.
    pub fn compress(&self) -> CompressedRistretto {
        let mut X = self.0.X;
        let mut Y = self.0.Y;
        let Z = &self.0.Z;
        let T = &self.0.T;

        let u1 = &(Z + &Y) * &(Z - &Y);
        let u2 = &X * &Y;
        // Ignore return value since this is always square
        let (_, invsqrt) = (&u1 * &u2.square()).invsqrt();
        let i1 = &invsqrt * &u1;
        let i2 = &invsqrt * &u2;
        let z_inv = &i1 * &(&i2 * T);
        let mut den_inv = i2;

        let iX = &X * &constants::SQRT_M1;
        let iY = &Y * &constants::SQRT_M1;
        let ristretto_magic = &constants::INVSQRT_A_MINUS_D;
        let enchanted_denominator = &i1 * ristretto_magic;

        let rotate = (T * &z_inv).is_negative();

        X.conditional_assign(&iY, rotate);
        Y.conditional_assign(&iX, rotate);
        den_inv.conditional_assign(&enchanted_denominator, rotate);

        Y.conditional_negate((&X * &z_inv).is_negative());

        let mut s = &den_inv * &(Z - &Y);
        let s_is_negative = s.is_negative();
        s.conditional_negate(s_is_negative);

        CompressedRistretto(s.to_bytes())
    }

    /// Double-and-compress a batch of points.  The Ristretto encoding
    /// is not batchable, since it requires an inverse square root.
    ///
    /// However, given input points \\( P\_1, \ldots, P\_n, \\)
    /// it is possible to compute the encodings of their doubles \\(
    /// \mathrm{enc}( [2]P\_1), \ldots, \mathrm{enc}( [2]P\_n ) \\)
    /// in a batch.
    ///
    /// ```
    /// # extern crate curve25519_dalek;
    /// # use curve25519_dalek::ristretto::RistrettoPoint;
    /// extern crate rand_core;
    /// use rand_core::OsRng;
    ///
    /// # // Need fn main() here in comment so the doctest compiles
    /// # // See https://doc.rust-lang.org/book/documentation.html#documentation-as-tests
    /// # fn main() {
    /// let mut rng = OsRng;
    /// let points: Vec<RistrettoPoint> =
    ///     (0..32).map(|_| RistrettoPoint::random(&mut rng)).collect();
    ///
    /// let compressed = RistrettoPoint::double_and_compress_batch(&points);
    ///
    /// for (P, P2_compressed) in points.iter().zip(compressed.iter()) {
    ///     assert_eq!(*P2_compressed, (P + P).compress());
    /// }
    /// # }
    /// ```
    #[cfg(feature = "alloc")]
    pub fn double_and_compress_batch<'a, I>(points: I) -> Vec<CompressedRistretto>
        where I: IntoIterator<Item = &'a RistrettoPoint>
    {
        #[derive(Copy, Clone, Debug)]
        struct BatchCompressState {
            e: FieldElement,
            f: FieldElement,
            g: FieldElement,
            h: FieldElement,
            eg: FieldElement,
            fh: FieldElement,
        }

        impl BatchCompressState {
            fn efgh(&self) -> FieldElement {
                &self.eg * &self.fh
            }
        }

        impl<'a> From<&'a RistrettoPoint> for BatchCompressState {
            fn from(P: &'a RistrettoPoint) -> BatchCompressState {
                let XX = P.0.X.square();
                let YY = P.0.Y.square();
                let ZZ = P.0.Z.square();
                let dTT = &P.0.T.square() * &constants::EDWARDS_D;

                let e = &P.0.X * &(&P.0.Y + &P.0.Y); // = 2*X*Y
                let f = &ZZ + &dTT;                  // = Z^2 + d*T^2
                let g = &YY + &XX;                   // = Y^2 - a*X^2
                let h = &ZZ - &dTT;                  // = Z^2 - d*T^2

                let eg = &e * &g;
                let fh = &f * &h;

                BatchCompressState{ e, f, g, h, eg, fh }
            }
        }

        let states: Vec<BatchCompressState> = points.into_iter().map(BatchCompressState::from).collect();

        let mut invs: Vec<FieldElement> = states.iter().map(|state| state.efgh()).collect();

        FieldElement::batch_invert(&mut invs[..]);

        states.iter().zip(invs.iter()).map(|(state, inv): (&BatchCompressState, &FieldElement)| {
            let Zinv = &state.eg * &inv;
            let Tinv = &state.fh * &inv;

            let mut magic = constants::INVSQRT_A_MINUS_D;

            let negcheck1 = (&state.eg * &Zinv).is_negative();

            let mut e = state.e;
            let mut g = state.g;
            let mut h = state.h;

            let minus_e = -&e;
            let f_times_sqrta = &state.f * &constants::SQRT_M1;

            e.conditional_assign(&state.g,       negcheck1);
            g.conditional_assign(&minus_e,       negcheck1);
            h.conditional_assign(&f_times_sqrta, negcheck1);

            magic.conditional_assign(&constants::SQRT_M1, negcheck1);

            let negcheck2 = (&(&h * &e) * &Zinv).is_negative();

            g.conditional_negate(negcheck2);

            let mut s = &(&h - &g) * &(&magic * &(&g * &Tinv));

            let s_is_negative = s.is_negative();
            s.conditional_negate(s_is_negative);

            CompressedRistretto(s.to_bytes())
        }).collect()
    }


    /// Return the coset self + E[4], for debugging.
    fn coset4(&self) -> [EdwardsPoint; 4] {
        [  self.0
        , &self.0 + &constants::EIGHT_TORSION[2]
        , &self.0 + &constants::EIGHT_TORSION[4]
        , &self.0 + &constants::EIGHT_TORSION[6]
        ]
    }

    /// Computes the Ristretto Elligator map.
    ///
    /// # Note
    ///
    /// This method is not public because it's just used for hashing
    /// to a point -- proper elligator support is deferred for now.
    pub(crate) fn elligator_ristretto_flavor(r_0: &FieldElement) -> RistrettoPoint {
        let i = &constants::SQRT_M1;
        let d = &constants::EDWARDS_D;
        let one_minus_d_sq = &constants::ONE_MINUS_EDWARDS_D_SQUARED;
        let d_minus_one_sq = &constants::EDWARDS_D_MINUS_ONE_SQUARED;
        let mut c = constants::MINUS_ONE;

        let one = FieldElement::one();

        let r = i * &r_0.square();
        let N_s = &(&r + &one) * &one_minus_d_sq;
        let D = &(&c - &(d * &r)) * &(&r + d);

        let (Ns_D_is_sq, mut s) = FieldElement::sqrt_ratio_i(&N_s, &D);
        let mut s_prime = &s * r_0;
        let s_prime_is_pos = !s_prime.is_negative();
        s_prime.conditional_negate(s_prime_is_pos);

        s.conditional_assign(&s_prime, !Ns_D_is_sq);
        c.conditional_assign(&r, !Ns_D_is_sq);

        let N_t = &(&(&c * &(&r - &one)) * &d_minus_one_sq) - &D;
        let s_sq = s.square();

        use backend::serial::curve_models::CompletedPoint;

        // The conversion from W_i is exactly the conversion from P1xP1.
        RistrettoPoint(CompletedPoint{
            X: &(&s + &s) * &D,
            Z: &N_t * &constants::SQRT_AD_MINUS_ONE,
            Y: &FieldElement::one() - &s_sq,
            T: &FieldElement::one() + &s_sq,
        }.to_extended())
    }


    /// Hash a slice of bytes into a `RistrettoPoint`.
    ///
    /// Takes a type parameter `D`, which is any `Digest` producing 64
    /// bytes of output.
    ///
    /// Convenience wrapper around `from_hash`.
    ///
    /// # Implementation
    ///
    /// Uses the Ristretto-flavoured Elligator 2 map, so that the
    /// discrete log of the output point with respect to any other
    /// point should be unknown.  The map is applied twice and the
    /// results are added, to ensure a uniform distribution.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate curve25519_dalek;
    /// # use curve25519_dalek::ristretto::RistrettoPoint;
    /// extern crate sha2;
    /// use sha2::Sha512;
    ///
    /// # // Need fn main() here in comment so the doctest compiles
    /// # // See https://doc.rust-lang.org/book/documentation.html#documentation-as-tests
    /// # fn main() {
    /// let msg = "To really appreciate architecture, you may even need to commit a murder";
    /// let P = RistrettoPoint::hash_from_bytes::<Sha512>(msg.as_bytes());
    /// # }
    /// ```
    ///
    pub fn hash_from_bytes<D>(input: &[u8]) -> RistrettoPoint
        where D: Digest<OutputSize = U64> + Default
    {
        let mut hash = D::default();
        hash.update(input);
        RistrettoPoint::from_hash(hash)
    }

    /// Construct a `RistrettoPoint` from an existing `Digest` instance.
    ///
    /// Use this instead of `hash_from_bytes` if it is more convenient
    /// to stream data into the `Digest` than to pass a single byte
    /// slice.
    pub fn from_hash<D>(hash: D) -> RistrettoPoint
        where D: Digest<OutputSize = U64> + Default
    {
        // dealing with generic arrays is clumsy, until const generics land
        let output = hash.finalize();
        let mut output_bytes = [0u8; 64];
        output_bytes.copy_from_slice(&output.as_slice());

        RistrettoPoint::from_uniform_bytes(&output_bytes)
    }

    /// Construct a `RistrettoPoint` from 64 bytes of data.
    ///
    /// If the input bytes are uniformly distributed, the resulting
    /// point will be uniformly distributed over the group, and its
    /// discrete log with respect to other points should be unknown.
    ///
    /// # Implementation
    ///
    /// This function splits the input array into two 32-byte halves,
    /// takes the low 255 bits of each half mod p, applies the
    /// Ristretto-flavored Elligator map to each, and adds the results.
    pub fn from_uniform_bytes(bytes: &[u8; 64]) -> RistrettoPoint {
        let mut r_1_bytes = [0u8; 32];
        r_1_bytes.copy_from_slice(&bytes[0..32]);
        let r_1 = FieldElement::from_bytes(&r_1_bytes);
        let R_1 = RistrettoPoint::elligator_ristretto_flavor(&r_1);

        let mut r_2_bytes = [0u8; 32];
        r_2_bytes.copy_from_slice(&bytes[32..64]);
        let r_2 = FieldElement::from_bytes(&r_2_bytes);
        let R_2 = RistrettoPoint::elligator_ristretto_flavor(&r_2);

        // Applying Elligator twice and adding the results ensures a
        // uniform distribution.
        &R_1 + &R_2
    }
}

impl Identity for RistrettoPoint {
    fn identity() -> RistrettoPoint {
        RistrettoPoint(EdwardsPoint::identity())
    }
}

impl Default for RistrettoPoint {
    fn default() -> RistrettoPoint {
        RistrettoPoint::identity()
    }
}

// ------------------------------------------------------------------------
// Equality
// ------------------------------------------------------------------------

impl PartialEq for RistrettoPoint {
    fn eq(&self, other: &RistrettoPoint) -> bool {
        self.ct_eq(other).unwrap_u8() == 1u8
    }
}

impl ConstantTimeEq for RistrettoPoint {
    /// Test equality between two `RistrettoPoint`s.
    ///
    /// # Returns
    ///
    /// * `Choice(1)` if the two `RistrettoPoint`s are equal;
    /// * `Choice(0)` otherwise.
    fn ct_eq(&self, other: &RistrettoPoint) -> Choice {
        let X1Y2 = &self.0.X * &other.0.Y;
        let Y1X2 = &self.0.Y * &other.0.X;
        let X1X2 = &self.0.X * &other.0.X;
        let Y1Y2 = &self.0.Y * &other.0.Y;

        X1Y2.ct_eq(&Y1X2) | X1X2.ct_eq(&Y1Y2)
    }
}

impl Eq for RistrettoPoint {}

// ------------------------------------------------------------------------
// Arithmetic
// ------------------------------------------------------------------------

impl<'a, 'b> Add<&'b RistrettoPoint> for &'a RistrettoPoint {
    type Output = RistrettoPoint;

    fn add(self, other: &'b RistrettoPoint) -> RistrettoPoint {
        RistrettoPoint(&self.0 + &other.0)
    }
}

define_add_variants!(LHS = RistrettoPoint, RHS = RistrettoPoint, Output = RistrettoPoint);

impl<'b> AddAssign<&'b RistrettoPoint> for RistrettoPoint {
    fn add_assign(&mut self, _rhs: &RistrettoPoint) {
        *self = (self as &RistrettoPoint) + _rhs;
    }
}

define_add_assign_variants!(LHS = RistrettoPoint, RHS = RistrettoPoint);

impl<'a, 'b> Sub<&'b RistrettoPoint> for &'a RistrettoPoint {
    type Output = RistrettoPoint;

    fn sub(self, other: &'b RistrettoPoint) -> RistrettoPoint {
        RistrettoPoint(&self.0 - &other.0)
    }
}

define_sub_variants!(LHS = RistrettoPoint, RHS = RistrettoPoint, Output = RistrettoPoint);

impl<'b> SubAssign<&'b RistrettoPoint> for RistrettoPoint {
    fn sub_assign(&mut self, _rhs: &RistrettoPoint) {
        *self = (self as &RistrettoPoint) - _rhs;
    }
}

define_sub_assign_variants!(LHS = RistrettoPoint, RHS = RistrettoPoint);

impl<T> Sum<T> for RistrettoPoint
where
    T: Borrow<RistrettoPoint>
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>
    {
        iter.fold(RistrettoPoint::identity(), |acc, item| acc + item.borrow())
    }
}

impl<'a> Neg for &'a RistrettoPoint {
    type Output = RistrettoPoint;

    fn neg(self) -> RistrettoPoint {
        RistrettoPoint(-&self.0)
    }
}

impl Neg for RistrettoPoint {
    type Output = RistrettoPoint;

    fn neg(self) -> RistrettoPoint {
        -&self
    }
}

impl<'b> MulAssign<&'b Scalar> for RistrettoPoint {
    fn mul_assign(&mut self, scalar: &'b Scalar) {
        let result = (self as &RistrettoPoint) * scalar;
        *self = result;
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a RistrettoPoint {
    type Output = RistrettoPoint;
    /// Scalar multiplication: compute `scalar * self`.
    fn mul(self, scalar: &'b Scalar) -> RistrettoPoint {
        RistrettoPoint(self.0 * scalar)
    }
}

impl<'a, 'b> Mul<&'b RistrettoPoint> for &'a Scalar {
    type Output = RistrettoPoint;

    /// Scalar multiplication: compute `self * scalar`.
    fn mul(self, point: &'b RistrettoPoint) -> RistrettoPoint {
        RistrettoPoint(self * point.0)
    }
}

define_mul_assign_variants!(LHS = RistrettoPoint, RHS = Scalar);

define_mul_variants!(LHS = RistrettoPoint, RHS = Scalar, Output = RistrettoPoint);
define_mul_variants!(LHS = Scalar, RHS = RistrettoPoint, Output = RistrettoPoint);

// ------------------------------------------------------------------------
// Multiscalar Multiplication impls
// ------------------------------------------------------------------------

// These use iterator combinators to unwrap the underlying points and
// forward to the EdwardsPoint implementations.

#[cfg(feature = "alloc")]
impl MultiscalarMul for RistrettoPoint {
    type Point = RistrettoPoint;

    fn multiscalar_mul<I, J>(scalars: I, points: J) -> RistrettoPoint
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<RistrettoPoint>,
    {
        let extended_points = points.into_iter().map(|P| P.borrow().0);
        RistrettoPoint(
            EdwardsPoint::multiscalar_mul(scalars, extended_points)
        )
    }
}

#[cfg(feature = "alloc")]
impl VartimeMultiscalarMul for RistrettoPoint {
    type Point = RistrettoPoint;

    fn optional_multiscalar_mul<I, J>(scalars: I, points: J) -> Option<RistrettoPoint>
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator<Item = Option<RistrettoPoint>>,
    {
        let extended_points = points.into_iter().map(|opt_P| opt_P.map(|P| P.borrow().0));

        EdwardsPoint::optional_multiscalar_mul(scalars, extended_points).map(RistrettoPoint)
    }
}

/// Precomputation for variable-time multiscalar multiplication with `RistrettoPoint`s.
// This wraps the inner implementation in a facade type so that we can
// decouple stability of the inner type from the stability of the
// outer type.
#[cfg(feature = "alloc")]
pub struct VartimeRistrettoPrecomputation(scalar_mul::precomputed_straus::VartimePrecomputedStraus);

#[cfg(feature = "alloc")]
impl VartimePrecomputedMultiscalarMul for VartimeRistrettoPrecomputation {
    type Point = RistrettoPoint;

    fn new<I>(static_points: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Self::Point>,
    {
        Self(
            scalar_mul::precomputed_straus::VartimePrecomputedStraus::new(
                static_points.into_iter().map(|P| P.borrow().0),
            ),
        )
    }

    fn optional_mixed_multiscalar_mul<I, J, K>(
        &self,
        static_scalars: I,
        dynamic_scalars: J,
        dynamic_points: K,
    ) -> Option<Self::Point>
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<Scalar>,
        K: IntoIterator<Item = Option<Self::Point>>,
    {
        self.0
            .optional_mixed_multiscalar_mul(
                static_scalars,
                dynamic_scalars,
                dynamic_points.into_iter().map(|P_opt| P_opt.map(|P| P.0)),
            )
            .map(RistrettoPoint)
    }
}

impl RistrettoPoint {
    /// Compute \\(aA + bB\\) in variable time, where \\(B\\) is the
    /// Ristretto basepoint.
    pub fn vartime_double_scalar_mul_basepoint(
        a: &Scalar,
        A: &RistrettoPoint,
        b: &Scalar,
    ) -> RistrettoPoint {
        RistrettoPoint(
            EdwardsPoint::vartime_double_scalar_mul_basepoint(a, &A.0, b)
        )
    }
}

/// A precomputed table of multiples of a basepoint, used to accelerate
/// scalar multiplication.
///
/// A precomputed table of multiples of the Ristretto basepoint is
/// available in the `constants` module:
/// ```
/// use curve25519_dalek::constants;
/// use curve25519_dalek::scalar::Scalar;
///
/// let a = Scalar::from(87329482u64);
/// let P = &a * &constants::RISTRETTO_BASEPOINT_TABLE;
/// ```
#[derive(Clone)]
pub struct RistrettoBasepointTable(pub(crate) EdwardsBasepointTable);

impl<'a, 'b> Mul<&'b Scalar> for &'a RistrettoBasepointTable {
    type Output = RistrettoPoint;

    fn mul(self, scalar: &'b Scalar) -> RistrettoPoint {
        RistrettoPoint(&self.0 * scalar)
    }
}

impl<'a, 'b> Mul<&'a RistrettoBasepointTable> for &'b Scalar {
    type Output = RistrettoPoint;

    fn mul(self, basepoint_table: &'a RistrettoBasepointTable) -> RistrettoPoint {
        RistrettoPoint(self * &basepoint_table.0)
    }
}

impl RistrettoBasepointTable {
    /// Create a precomputed table of multiples of the given `basepoint`.
    pub fn create(basepoint: &RistrettoPoint) -> RistrettoBasepointTable {
        RistrettoBasepointTable(EdwardsBasepointTable::create(&basepoint.0))
    }

    /// Get the basepoint for this table as a `RistrettoPoint`.
    pub fn basepoint(&self) -> RistrettoPoint {
        RistrettoPoint(self.0.basepoint())
    }
}

// ------------------------------------------------------------------------
// Constant-time conditional selection
// ------------------------------------------------------------------------

impl ConditionallySelectable for RistrettoPoint {
    /// Conditionally select between `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate subtle;
    /// # extern crate curve25519_dalek;
    /// #
    /// use subtle::ConditionallySelectable;
    /// use subtle::Choice;
    /// #
    /// # use curve25519_dalek::traits::Identity;
    /// # use curve25519_dalek::ristretto::RistrettoPoint;
    /// # use curve25519_dalek::constants;
    /// # fn main() {
    ///
    /// let A = RistrettoPoint::identity();
    /// let B = constants::RISTRETTO_BASEPOINT_POINT;
    ///
    /// let mut P = A;
    ///
    /// P = RistrettoPoint::conditional_select(&A, &B, Choice::from(0));
    /// assert_eq!(P, A);
    /// P = RistrettoPoint::conditional_select(&A, &B, Choice::from(1));
    /// assert_eq!(P, B);
    /// # }
    /// ```
    fn conditional_select(
        a: &RistrettoPoint,
        b: &RistrettoPoint,
        choice: Choice,
    ) -> RistrettoPoint {
        RistrettoPoint(EdwardsPoint::conditional_select(&a.0, &b.0, choice))
    }
}

// ------------------------------------------------------------------------
// Debug traits
// ------------------------------------------------------------------------

impl Debug for CompressedRistretto {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "CompressedRistretto: {:?}", self.as_bytes())
    }
}

impl Debug for RistrettoPoint {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        let coset = self.coset4();
        write!(f, "RistrettoPoint: coset \n{:?}\n{:?}\n{:?}\n{:?}",
               coset[0], coset[1], coset[2], coset[3])
    }
}

// ------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use scalar::Scalar;
    use constants;
    use edwards::CompressedEdwardsY;
    use traits::{Identity};
    use super::*;

}
