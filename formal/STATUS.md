# Formal evidence status

`DecisionGuards.lean` contains nine proof declarations for scaled-integer
acceptance/rejection guards, monotone refinement, chronological function
composition, deterministic replay, and publication guards. It imports `Std`
and contains no `sorry`, `admit`, or new axioms. **It has not been compiled.**
The authoring environment had no Lean executable; attempts to obtain a binary
were unsuccessful. Source inspection is not machine-checked evidence.

Run `bash scripts/check_lean.sh` in an environment with the pinned toolchain.
The script fails rather than substituting a passing placeholder. Resolve any
elaboration errors before changing this status. The toolchain version is a
reproducibility target, not a claim that this source was validated with it.

Even a successful build would establish only these abstract lemmas. It would
not prove the real-valued exponential transport theorem, the rational
representation bridge, the actual proposal law, numerical enclosures, the
compiler lowering, CUDA race freedom, or the engine's state restoration.
The full mathematical derivations and their assumptions are in the paper.
