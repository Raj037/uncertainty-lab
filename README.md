# Uncertainty-Lab: Bayesian, Metrological & Imprecise-Probability UQ Engine

[![Release: v1.0.0](https://img.shields.io/badge/release-v1.0.0-blue.svg)](RELEASE_NOTES_v1.0.0.md)
[![GUM-5 Benchmarks](https://img.shields.io/badge/JCGM_GUM--5-16%2F16_Verified-brightgreen.svg)](skills/uncertainty-propagation/references/gum5-2026-benchmarks.md)
[![ArviZ Cross-Check](https://img.shields.io/badge/Posterior_Diagnostics-ArviZ_Validated-orange.svg)](skills/uncertainty-propagation/references/posterior-arviz-crosscheck.json)
[![Integrity](https://img.shields.io/badge/SHA--256-Manifest_Verified-success.svg)](SOURCE_MANIFEST.sha256)

A production-grade scientific computing framework and agentic skill suite for **rigorous uncertainty quantification (UQ)**, **Bayesian posterior inference**, **multi-fidelity surrogate modeling**, and **structural reliability analysis**. 

Designed to eliminate silent numerical failures in scientific and quantitative pipelines, `uncertainty-lab` enforces strict mathematical invariants, **BIPM/JCGM GUM-5 metrological compliance**, and cryptographically signed adversarial release gates.

---

## System Architecture

```mermaid
graph TD
    A[Input System / Physical Model] --> B{Propagation & Inference Layer}
    
    B -->|Gradient / Exact| C[Autodiff & Taylor Series Engine]
    B -->|Sampling / MCMC| D[Adaptive MC / HMC-NUTS / Constrained NUTS]
    B -->|Imprecise / Tail-Risk| E[Probability Boxes P-Box & Non-Gaussian Reliability]
    B -->|High-Cost Compute| F[Multi-Fidelity & Multi-Output Surrogates]
    
    C & D & E & F --> G[Diagnostics & Calibration Layer]
    G --> H[ArviZ Posterior Cross-Check & Conformal Calibration]
    H --> I[SHA-256 Validated Release Gate Receipt]
