# Native proof runs

`discover_batch` can make multiple real-primary calls and must not be run inline without a strict external timeout. Run multi-stage native proofs detached with an explicit bounded wrapper, then sleep and inspect the log. Use focused unit tests or a single bounded primary call for inline verification.
