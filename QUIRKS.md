# Explicit choices for experiment 1

1. **Exactly offsets −1, 0, +1**, including the top1 anchor: at most three
   tokens total, not three additional tokens. Version 2 removes +2 from the
   original four-token version. Neighbor means token index, not
   word or nearest remaining MASK. Positive offsets point right.
2. **Only still-masked eligible slots.** Already-unmasked slots are skipped.
   We interpret the user's “if applicable” this way; overwriting revealed
   tokens would be a different experiment. No replacement of skipped slots.
3. **No neighbor-confidence threshold.** Even a low-confidence neighbor is
   committed. This is the proposed ungated test, not a threshold variant.
4. **One simultaneous batch from one forward.** We do not reveal the anchor,
   run the model again, and then pick neighbors. The logged anchor-first order
   is serialization only; there is no model call between commits.
5. **Same baseline checkpoint:** LLaDA-8B-Instruct, pinned model commit
   `08b83a6feb34df1a6011b80c3c00c7563e963b07`, BF16, no KV cache, no sampling,
   no classifier-free guidance. Top1 ties use the lower position index.
6. **Question-only user chat prompt**, with the checkpoint's chat template and
   assistant-generation marker. No added solving or answer-format instruction.
7. **EOS remains allowed and does not stop the loop early.** Both policies fill
   the whole response window (pilot 256 slots); displayed/scored text is cut
   before the final first EOS/end-of-turn token. This matches the previous
   baseline and avoids declaring a stop while earlier masks remain unresolved.
   Special-token neighbors are allowed too; their confidence is recorded.
8. **MASK cannot be selected as the predicted token.** Its raw probability is
   retained when computing confidence/entropy; other probabilities are not
   renormalized after excluding MASK from ranking.
9. **Full response is one block by default.** Optional block boundaries restrict
   the anchor and neighbors. The four-MASK statistical region definition does
   not influence this decoder.
10. **Matched new baseline.** The script reruns top1 on the same questions and
    pinned dataset commit as the window policy, rather than assuming old runs
    match the environment. Policy order is top1 then window; no warmup is added.
11. **Numeric scoring:** use a marked answer if present, otherwise the last
    numeric value; reference answers require the dataset's `####` marker. The
    marker-only score is retained as a diagnostic. This is not an external judge.
12. **Runtime is instrumented.** Per-question elapsed time includes scoring,
    logging, and serialization. Forward time is reported separately; total
    forwards include EOS/post-stop slots. Forced-neighbor statistics are shown
    for both all slots and nonspecial pre-final-stop answer positions.

Only pure CPU tests have run locally. GPU memory requirements, realized speed,
and accuracy are unverified until the Vast runs complete. Source is derived
from the existing experiment collector; this repository is intentionally
independent and retains the `confidence_geography` Python module name so the
collector and offline tooling need minimal changes.
