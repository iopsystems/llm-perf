//! Replay of `llm-perf convert-trace` sessions against a server. See
//! `docs/superpowers/specs/2026-10-02-trace-replay-design.md`.

pub mod filler;
pub mod kv_events;
pub mod kv_residency;
pub mod prompt;
pub mod render;
pub mod runner;
pub mod sample;
pub mod schedule;
pub mod server;
