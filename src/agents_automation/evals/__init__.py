"""The scenario eval harness.

Interview scenarios: simulated users talk to the real interview loop; each run is checked by
deterministic checks and then by a rubric judge from a *different* model family. Role
scenarios: the synthetic office is built with the configured system and one situation of the
knowledge, summarizer, routing or interviewer guards is checked. Every scenario runs at least
twice and the runs are classified PASS / FLAKY / FAIL / VOID. Safety scenarios are hard gates.
Mutants (deliberately disabled guards) prove the harness actually catches regressions.
"""
