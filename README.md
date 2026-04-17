# CHB-MIT DeltaPhi Test

This project explores whether a simple deterministic ΔΦ operator
can highlight preictal instability in EEG signals using the CHB-MIT dataset.

## Method
The pipeline computes three interpretable features:
- S (Structure): variance and signal range
- I (Information): Shannon entropy
- C (Coupling): inter-channel correlation

These are combined into a scalar instability measure ΔΦ.

## Features
- S/I/C feature extraction
- ΔΦ computation
- EEG window-based analysis (10s windows)

## Status
This is an exploratory implementation.
No full patient-wise validation has been performed yet.

## Goal
To test whether a simple, deterministic instability operator
can serve as an early-warning signal in EEG data.

## Next Steps
- Full CHB-MIT dataset evaluation
- Patient-wise cross-validation
- Comparison against baseline features
