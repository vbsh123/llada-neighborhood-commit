"""Exact row-update accounting and explicitly approximate FLOP estimates."""
from .protocol import layer_intervals


def refresh_count(forwards, interval):
    if forwards < 1 or interval < 0:
        raise ValueError('Positive forwards and nonnegative interval required')
    return 1 if interval == 0 else 1 + (forwards - 1) // interval


def row_budget(prepared, length, forwards, specification, layers):
    if specification.get('pruning'):
        raise ValueError('Pruning requires measured execution accounting, not a cache-only row budget')
    total = prepared.answer_start + length
    active = [total * forwards] * layers
    claimed = set()
    for group in specification['schedule']:
        positions = set(prepared.group_positions(group['group']))
        if claimed & positions:
            raise ValueError('Overlapping cache groups')
        claimed.update(positions)
        for layer, interval in enumerate(layer_intervals(group, layers)):
            active[layer] -= len(positions) * (forwards - refresh_count(forwards, interval))
    return {'forwards': forwards, 'sequence_rows': total,
            'computed_rows_by_layer': active,
            'computed_token_layers': sum(active),
            'dense_token_layers': total * forwards * layers,
            'refreshes_by_group': {s['group']: [refresh_count(forwards, r) for r in
                layer_intervals(s, layers)] for s in specification['schedule']}}


def flop_estimate(llm, budget):
    """Linear GEMMs + QK/AV only; no latency bound or hardware model."""
    from .hooks import blocks_of
    blocks = blocks_of(llm)
    active = budget['computed_rows_by_layer']
    projection = 0
    for block, rows in zip(blocks, active):
        names = ('q_proj', 'k_proj', 'v_proj', 'attn_out', 'ff_proj', 'up_proj', 'ff_out')
        projection += 2 * rows * sum(getattr(block, n).weight.numel() for n in names)
    n, f = budget['sequence_rows'], budget['forwards']
    attention = 4 * n * llm.config.d_model * sum(active)
    transformer = llm.model.transformer
    head = transformer['wte'] if llm.config.weight_tying else transformer['ff_out']
    vocabulary = 2 * n * f * head.weight.numel()
    return {'projection_flops': projection, 'qk_av_flops': attention,
            'vocabulary_flops': vocabulary, 'total_estimated_flops': projection + attention + vocabulary,
            'note': 'Arithmetic estimate, NOT a wall-time upper bound. Omits norms, activations, softmax, '
                    'embeddings, memory traffic, cache copies, launch overhead and decoder bookkeeping. '
                    'Full-sequence vocabulary projection and every KV column remain present.'}
