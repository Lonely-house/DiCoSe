# Working on arb-stem-mss

For data preparation, DET training, evaluation, or inference, read
`skills/arb-stem-det/SKILL.md` and the relevant linked references first. Keep
commands and documented artifacts aligned with the actual CLI.

- The current phase is text-conditioned DET using the real vendored DiCoSe
  U-Net. Preserve the vendored bytes and license; implement adaptations in the
  wrapper. Record deliberate upstream changes in `UPSTREAM.json`.
- Only the mixture and query embedding enter the model. Ground-truth target
  and submixture are supervision. Same-label tracks are grouped unless an
  explicit instance query and matching annotation exist.
- Use song-disjoint splits. Preserve the official MUSDB and Slakh test
  boundaries. Do not substitute mastered mixture audio for the documented sum
  of stems without introducing an explicit new experiment.
- Mock embeddings and generated tones are engineering fixtures. Keep the
  explicit mock gates and checkpoint labels; never report them as language or
  music-quality evaluation.
- Run relevant tests with `python -m pytest`. The complete synthetic workflow
  is `python scripts/smoke_reproduction.py --work-dir artifacts/smoke_new`.
  Keep datasets, audio, embeddings, checkpoints, and runs out of Git.
- DIF/CD are future work. Do not invent completed training stages or measured
  quality gains. Update `docs/reproduction-status.md` only with actual evidence.
