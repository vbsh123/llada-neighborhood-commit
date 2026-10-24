"""Fixed-quota DIFFA decoding; full-canvas forwards and safe EOS stopping."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Prepared:
    # Discrete IDs exclude audio, just as in upstream origin_generate.py.
    prompt: torch.Tensor
    speech: torch.Tensor
    insertion: int
    acoustic_count: int
    mask_id: int = 126336

    @property
    def audio_count(self):
        return self.speech.shape[1]

    @property
    def answer_start(self):
        return self.prompt.shape[1] + self.audio_count

    @property
    def audio_positions(self):
        return list(range(self.insertion, self.insertion + self.audio_count))

    def group_positions(self, group):
        if group == "prompt":
            return list(range(self.insertion)) + list(range(self.insertion + self.audio_count, self.answer_start))
        if group == "prefix":
            return list(range(self.answer_start))
        if group == "acoustic":
            return self.audio_positions[:self.acoustic_count]
        if group == "semantic":
            return self.audio_positions[self.acoustic_count:]
        if group == "both":
            return self.audio_positions
        raise ValueError(f"Unknown audio group: {group}")


def schedule(length, block_length, steps):
    if min(length, block_length, steps) <= 0 or length % block_length:
        raise ValueError("Positive length must be divisible by block length")
    blocks = length // block_length
    if steps % blocks or steps > length:
        raise ValueError("Steps must divide evenly across blocks and be <= answer length")
    per_block = steps // blocks
    base, remainder = divmod(block_length, per_block)
    return [(block, local, base + (local < remainder))
            for block in range(blocks) for local in range(per_block)]


def forward_logits(llm, prepared, x):
    text = llm.get_input_embeddings()(x)
    embeddings = torch.cat((text[:, :prepared.insertion], prepared.speech.to(text.dtype),
                            text[:, prepared.insertion:]), dim=1)
    # Clone answer slice so the full-sequence logits storage can be freed.
    return llm(inputs_embeds=embeddings, use_cache=False).logits[:, prepared.answer_start:].clone()


def commit(x, logits, prepared, block, block_length, count):
    answer = x[:, prepared.prompt.shape[1]:]
    eligible = answer == prepared.mask_id
    # Matches upstream: earlier unresolved slots remain eligible; future blocks do not.
    eligible[:, (block + 1) * block_length:] = False
    if int(eligible.sum()) < count:
        raise ValueError("Insufficient masks for the fixed unmasking schedule")
    predictions = logits.argmax(-1)
    # Match upstream's float64 confidence, calculated only on answer rows.
    confidence = F.softmax(logits.double(), dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(eligible, confidence, -torch.inf)
    positions = torch.topk(confidence[0], k=count).indices
    before = answer.clone()
    answer[0, positions] = predictions[0, positions]
    return int((answer != before).sum()), positions.tolist()


@torch.no_grad()
def decode(llm, prepared, length, block_length, steps, progress_every=0, eos_id=None):
    plan = schedule(length, block_length, steps)
    x = prepared.prompt.new_full((1, prepared.prompt.shape[1] + length), prepared.mask_id)
    x[:, :prepared.prompt.shape[1]] = prepared.prompt
    events = []
    stopped_at_eos = False
    for step, (block, local, count) in enumerate(plan):
        answer = x[:, prepared.prompt.shape[1]:]
        remaining = int((answer == prepared.mask_id).sum())
        logits = forward_logits(llm, prepared, x)
        changes, positions = commit(x, logits, prepared, block, block_length, count)
        del logits
        events.append(dict(step=step, block=block, step_in_block=local,
                           remaining_masks=remaining, committed_positions=positions))
        if eos_id is not None:
            eos_positions = (answer[0] == eos_id).nonzero().flatten()
            if len(eos_positions):
                first_eos = int(eos_positions[0])
                if not (answer[0, :first_eos] == prepared.mask_id).any():
                    stopped_at_eos = True
                    break
        if progress_every and (step == 0 or (step + 1) % progress_every == 0 or step + 1 == len(plan)):
            print(f"  forward {step + 1}/{len(plan)}; block {block + 1}; "
                  f"{remaining - changes} answer masks remain", flush=True)
    if not stopped_at_eos and (x[:, prepared.prompt.shape[1]:] == prepared.mask_id).any():
        raise ValueError("Decoder finished with unresolved masks")
    return x, events
