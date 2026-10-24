"""Global query scoring, reusing the verified physical-compaction machinery."""
import math

import torch

from experiments.audio_drift.progressive_pruning import ProgressivePruning


class GlobalPruning(ProgressivePruning):
    def local_queries(self):
        if self.spec['query_scope'] == 'all':
            return torch.arange(len(self.monitor_positions), device=self.monitor_positions.device)
        return super().local_queries()

    def observe_input(self, x):
        super().observe_input(x)
        if not self.monitoring:
            return
        scope = self.spec['query_scope']
        if scope == 'all':
            # Include every surviving prefix AND answer query, committed or MASK.
            # Select again per layer because deep-only pruning changes row maps.
            self.query_positions = None
        elif scope == 'all_masked':
            answer = x[0, self.prepared.prompt.shape[1]:]
            self.query_positions = (answer == self.prepared.mask_id).nonzero().flatten() + self.prepared.answer_start
        elif scope != 'current_block':
            raise ValueError(f'Unknown query scope: {scope}')

    def attention_scores(self, q, k):
        if self.spec['query_scope'] == 'all':
            local = torch.arange(q.shape[-2], device=q.device)
        else:
            local = self.query_local
        total = k.new_zeros(k.shape[-2], dtype=torch.float32)
        # Exactly average all selected query rows; chunking only bounds memory.
        # Sum over heads/queries before transferring anything to the CPU.
        key = k.float().transpose(-1, -2)
        for ids in local.split(self.spec.get('query_chunk', 32)):
            probability = (q.index_select(-2, ids).float() @ key / math.sqrt(q.shape[-1])).softmax(-1)
            total += probability.sum((0, 1, 2))
        return total / (q.shape[0] * q.shape[1] * len(local))

    def decide(self, scores, relative_scores=None):
        if self.query_positions is None:
            # Parent's decision record expects concrete positions. They are only
            # metadata: all-layer scoring above already included every query.
            self.query_positions = self.monitor_positions
            try:
                super().decide(scores, relative_scores)
            finally:
                self.query_positions = None
        else:
            super().decide(scores, relative_scores)

    def accounting(self):
        result = super().accounting()
        result['pruning']['query_scope'] = self.spec['query_scope']
        result['pruning']['note'] = (
            'Scores average all heads and all layers, with query_scope explicitly recorded. '
            'all includes every surviving prefix/answer query, including committed and future MASK positions. '
            'all_masked includes all remaining answer MASK queries across the entire canvas. '
            'Scores are actual softmax probabilities over all retained keys, including self-attention; '
            'threshold compares per-key mean with uniform attention at that layer. '
            'Decisions apply next forward; physical removal and original RoPE coordinates are preserved.')
        return result
