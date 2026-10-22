"""Deterministic, uncached DIFFA block decoding and paired continuations."""
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
        if group == "audio_matched":
            count = min(self.prompt.shape[1], self.audio_count)
            # Fixed, reproducible, evenly spread subset. Match prompt token count
            # when enough audio tokens exist; never include answer positions.
            return [self.insertion + i * self.audio_count // count for i in range(count)]
        if group == "acoustic":
            return self.audio_positions[:self.acoustic_count]
        if group == "semantic":
            return self.audio_positions[self.acoustic_count:]
        if group == "both":
            return self.audio_positions
        raise ValueError(f"Unknown audio group: {group}")

    def labels(self, length):
        groups = ["prompt"] * self.insertion
        groups += ["audio_acoustic"] * self.acoustic_count
        groups += ["audio_semantic"] * (self.audio_count - self.acoustic_count)
        groups += ["prompt"] * (self.prompt.shape[1] - self.insertion)
        return groups + ["answer"] * length

    def ids_with_audio(self, x):
        audio = x.new_full((1, self.audio_count), -1)
        return torch.cat((x[:, :self.insertion], audio, x[:, self.insertion:]), dim=1)


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


def commit(x, logits, prepared, block, block_length, count, priority_bonus=None):
    answer = x[:, prepared.prompt.shape[1]:]
    eligible = answer == prepared.mask_id
    # Matches upstream: earlier unresolved slots remain eligible; future blocks do not.
    eligible[:, (block + 1) * block_length:] = False
    if int(eligible.sum()) < count:
        raise ValueError("Insufficient masks for the fixed unmasking schedule")
    predictions = logits.argmax(-1)
    # Match upstream's float64 confidence, calculated only on answer rows.
    confidence = F.softmax(logits.double(), dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    if priority_bonus is not None:
        if priority_bonus.shape != confidence.shape or not torch.isfinite(priority_bonus).all():
            raise ValueError('Invalid commitment priority bonus')
        confidence = confidence + priority_bonus.to(confidence)
    confidence = torch.where(eligible, confidence, -torch.inf)
    positions = torch.topk(confidence[0], k=count).indices
    before = answer.clone()
    answer[0, positions] = predictions[0, positions]
    return int((answer != before).sum()), positions.tolist()


@torch.no_grad()
def decode(llm, prepared, length, block_length, steps, recorder=None, layers=None,
           start_x=None, start_step=0, first_logits=None, save_states=True,
           progress_every=0, eos_id=None, priority=None, progress_label='baseline'):
    plan = schedule(length, block_length, steps)
    if start_x is None:
        x = prepared.prompt.new_full((1, prepared.prompt.shape[1] + length), prepared.mask_id)
        x[:, :prepared.prompt.shape[1]] = prepared.prompt
    else:
        x = start_x.clone()
    states, events = [], []
    last_changes = 0
    stopped_at_eos = False
    for step in range(start_step, len(plan)):
        block, local, count = plan[step]
        answer = x[:, prepared.prompt.shape[1]:]
        remaining = int((answer == prepared.mask_id).sum())
        active_remaining = int((answer[:, block * block_length:(block + 1) * block_length]
                                == prepared.mask_id).sum())
        meta = {"step": step, "block": block, "step_in_block": local,
                "block_boundary": local == 0, "answer_length": length,
                "block_length": block_length, "remaining_masks": remaining,
                "remaining_fraction": remaining / length,
                "active_remaining_masks": active_remaining,
                "active_remaining_fraction": active_remaining / block_length,
                "previous_changed_tokens": last_changes}
        if save_states:
            states.append(x.detach().cpu().clone())
        if recorder is not None:
            recorder.begin(meta, prepared.ids_with_audio(x))
        if priority is not None:
            if first_logits is not None:raise ValueError('Priority scoring requires its own observed forward')
            priority.begin(x,step,block,block_length)
        logits = first_logits if step == start_step and first_logits is not None else forward_logits(llm, prepared, x)
        if recorder is not None:
            recorder.end(layers)
        bonus = priority.bonus(logits) if priority is not None else None
        last_changes, positions = commit(x, logits, prepared, block, block_length, count,bonus)
        del logits
        events.append({**meta, "changed_tokens": last_changes, "committed_positions": positions})
        if eos_id is not None:
            eos_positions = (answer[0] == eos_id).nonzero().flatten()
            if len(eos_positions):
                first_eos = int(eos_positions[0])
                if not (answer[0, :first_eos] == prepared.mask_id).any():
                    stopped_at_eos = True
                    break
        if progress_every and (step == start_step or (step + 1) % progress_every == 0 or step + 1 == len(plan)):
            print(f"  {progress_label} forward {step + 1}/{len(plan)}; block {block + 1}; "
                  f"{remaining - last_changes} answer masks remain", flush=True)
    if not stopped_at_eos and (x[:, prepared.prompt.shape[1]:] == prepared.mask_id).any():
        raise ValueError("Decoder finished with unresolved masks (upstream permits predicting MASK)")
    return x, states, events


def distribution_change(reference, changed, mask):
    """KL(reference || intervention) and top-1 changes at remaining masks.

    Both inputs live on CPU; chunking avoids float32 vocab-sized GPU buffers.
    """
    positions = mask.flatten().nonzero().flatten()
    if not len(positions):
        return {"count": 0, "kl_mean": None, "top1_change_rate": None}
    total_kl, total_changed = 0., 0
    for chunk in positions.split(8):
        a = reference[0, chunk].float().log_softmax(-1)
        b = changed[0, chunk].float().log_softmax(-1)
        total_kl += (a.exp() * (a - b)).sum().item()
        total_changed += int((a.argmax(-1) != b.argmax(-1)).sum())
    return {"count": len(positions), "kl_mean": max(0., total_kl / len(positions)),
            "top1_change_rate": total_changed / len(positions)}
