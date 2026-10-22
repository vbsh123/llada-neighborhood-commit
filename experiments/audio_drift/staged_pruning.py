"""Compact, staged prefix pruning without temporal KV/hidden banks.

Rows stay physically compact between transformer blocks. Absolute RoPE positions
are retained; only the last block restores sequence shape for the unchanged head.
Surviving representations are recomputed every forward.
"""
from contextlib import AbstractContextManager, nullcontext
import math

import torch

from .hooks import blocks_of


def staged_conditions(fractions=(.5, .75), acoustic_start=16, later_start=32, deep_layer=16, queries=16, seed=1729):
    if (not fractions or len(set(fractions)) != len(fractions) or
            any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions) or
            acoustic_start < 1 or later_start < acoustic_start or deep_layer < 1 or queries < 1):
        raise ValueError('Invalid staged pruning configuration')
    result = []
    for fraction in fractions:
        for variant, groups in [('a', ['acoustic']), ('as', ['acoustic', 'semantic']),
                                ('asp', ['acoustic', 'semantic', 'prompt'])]:
            stages = [dict(group=g, layer=0 if g == 'acoustic' else deep_layer,
                           start=acoustic_start if g == 'acoustic' else later_start,
                           fraction=fraction) for g in groups]
            for generic in (False, True):
                tag = format(fraction * 100, '.8g').replace('.', 'p')
                plan = dict(engine='compact_staged', stages=stages, generic=generic, queries=queries, seed=seed,
                            scope='prefix' if generic else variant, selection='attention',
                            start=acoustic_start, layer_start=0, fraction=fraction)
                stem=f'staged_{variant}_'
                suffix=f'_f{tag}_t{acoustic_start}x{later_start}_d{deep_layer}'
                result.append(dict(condition=stem+('generic' if generic else 'group')+suffix,
                                   paired_condition=stem+('group' if generic else 'generic')+suffix,
                                   mode='freeze', schedule=[dict(group='prefix', refresh_every=1)], pruning=plan))
    return result


class CompactStagedPruning(AbstractContextManager):
    def __init__(self, llm, prepared, specification):
        if llm.training:
            raise ValueError('Pruning requires eval mode')
        self.blocks = blocks_of(llm)
        self.prepared, self.spec = prepared, specification
        self.stages = specification['stages']
        if not self.stages or specification.get('engine') != 'compact_staged' or specification['queries'] < 1:
            raise ValueError('Invalid compact staged plan')
        for s in self.stages:
            if (s['group'] not in {'acoustic', 'semantic', 'prompt'} or
                    not 0 <= s['layer'] < len(self.blocks) or s['start'] < 1 or
                    not math.isfinite(s['fraction']) or not 0 <= s['fraction'] <= 1):
                raise ValueError('Invalid pruning stage')
        self.step, self.length = -1, None
        self.selected, self.records, self.patterns = {}, [], {}
        self.events, self.originals = [], []
        self.profile_scopes = False

    def scope(self, name):
        return torch.profiler.record_function(name) if self.profile_scopes else nullcontext()

    def pending_stages(self, layer):
        return [i for i,s in enumerate(self.stages) if i not in self.selected and
                s['layer'] == layer and self.step >= s['start']]

    def should_score(self, layer, pending):
        return bool(pending)

    def choose_pending(self, scores, positions, pending):
        return pending

    def drop_budget(self, stage):
        n = len(self.prepared.group_positions(stage['group']))
        return stage.get('drop_count', int(n*stage['fraction'])-int(n*stage.get('previous_fraction',0)))

    def known_removals(self, layer):
        return tuple(sorted({p for i, positions in self.selected.items()
                             if self.stages[i]['layer'] <= layer for p in positions}))

    def keep_rows(self, positions, removed):
        # Host position IDs are maintained alongside the device tensor. Never
        # transfer/synchronize the whole position map on each block/forward.
        if not removed:
            return None
        key = (self.position_ids, removed)
        if key not in self.patterns:
            omit = set(removed)
            local = [j for j,p in enumerate(self.position_ids) if p not in omit]
            ids = tuple(self.position_ids[j] for j in local)
            keep = None if ids == self.position_ids else torch.tensor(local,dtype=torch.long,device=positions.device)
            self.patterns[key] = (keep,ids)
        keep,ids = self.patterns[key]
        self.position_ids = ids
        return keep

    def __enter__(self):
        try:
            for layer, block in enumerate(self.blocks):
                if type(block).__name__ != 'LLaDALlamaBlock' or block._activation_checkpoint_fn is not None or not block.config.rope:
                    raise ValueError('Requires pinned uncheckpointed rotary LLaDALlamaBlock')
                existed = 'forward' in block.__dict__
                self.originals.append((block, existed, block.__dict__.get('forward')))
                original = block.forward

                def forward(x, attention_bias=None, layer_past=None, use_cache=False, replace_position=None,
                            _layer=layer, _block=block, _original=original):
                    if x.shape[0] != 1 or attention_bias is not None or layer_past is not None or use_cache or replace_position is not None:
                        raise ValueError('Pruning requires batch-one bidirectional no-past inference')
                    if _layer == 0:
                        self.step += 1
                        if self.length is None:
                            self.length = x.shape[1]
                            self.all_positions = torch.arange(self.length, device=x.device)
                            count = min(self.spec['queries'], self.length - self.prepared.answer_start)
                            self.probes = torch.linspace(self.prepared.answer_start, self.length-1, count,device=x.device).long().unique()
                        if x.shape[1] != self.length:
                            raise ValueError('Full input canvas must be preserved between forwards')
                        self.positions = self.all_positions
                        self.position_ids = tuple(range(self.length))
                        self.events.append(dict(step=self.step,total_rows=self.length,layer_qkv_rows=[],layer_computed_rows=[],layer_kv_columns=[]))
                    positions = self.positions
                    with self.scope('prune.compact_rows'):
                        keep = self.keep_rows(positions, self.known_removals(_layer))
                        if keep is not None:
                            x,positions = x.index_select(1,keep),positions.index_select(0,keep)
                    pending = self.pending_stages(_layer)
                    score_needed = self.should_score(_layer, pending)
                    event = self.events[-1]
                    qkv_rows = x.shape[1]
                    if not score_needed and x.shape[1] == self.length:
                        # Truly unchanged upstream execution; no clones/banks.
                        output = _original(x)
                        event['layer_qkv_rows'].append(qkv_rows)
                        event['layer_computed_rows'].append(qkv_rows)
                        event['layer_kv_columns'].append(qkv_rows)
                        self.positions = positions
                        return output
                    normalized = _block.attn_norm(x)
                    q,k,v = _block.q_proj(normalized),_block.k_proj(normalized),_block.v_proj(normalized)
                    if _block.q_norm is not None and _block.k_norm is not None:
                        q,k = _block.q_norm(q).to(k.dtype),_block.k_norm(k).to(k.dtype)
                    B,T,C = q.shape
                    q=q.view(B,T,_block.config.n_heads,C//_block.config.n_heads).transpose(1,2)
                    k=k.view(B,T,_block.config.effective_n_kv_heads,C//_block.config.n_heads).transpose(1,2)
                    v=v.view(B,T,_block.config.effective_n_kv_heads,C//_block.config.n_heads).transpose(1,2)
                    qr,kr=(q.float(),k.float()) if _block.config.rope_full_precision else (q,k)
                    rotary=_block.rotary_emb
                    with torch.autocast(q.device.type,enabled=False):
                        sin,cos=rotary.get_rotary_embedding(self.length,q.device)
                        sin,cos=sin.type_as(qr).index_select(-2,positions),cos.type_as(qr).index_select(-2,positions)
                        qr=rotary.apply_rotary_pos_emb(sin,cos,qr)
                        kr=rotary.apply_rotary_pos_emb(sin,cos,kr)
                    q,k=qr.type_as(q),kr.type_as(k)
                    if score_needed:
                        with self.scope('prune.selection'):
                            local=torch.searchsorted(positions,self.probes)
                            probability=(q[:,:,local,:].float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
                            scores=probability.mean((0,1,2))
                            pending=self.choose_pending(scores,positions,pending)
                            available=set(self.position_ids)
                            removed_now=set()
                            for i in pending:
                                stage=self.stages[i]
                                original_group=self.prepared.group_positions(stage['group'])
                                budget=self.drop_budget(stage)
                                pool=sorted(available.intersection(range(self.prepared.answer_start) if self.spec['generic'] else original_group)-removed_now)
                                if budget>len(pool):
                                    raise ValueError('Requested stage budget exceeds its available pool')
                                candidate=torch.tensor(pool,dtype=torch.long,device=x.device)
                                candidate_local=torch.searchsorted(positions,candidate)
                                chosen=torch.argsort(scores.index_select(0,candidate_local),stable=True)[:budget]
                                dropped=candidate.index_select(0,chosen)
                                chosen_local=candidate_local.index_select(0,chosen)
                                absolute=tuple(sorted(dropped.detach().cpu().tolist()))
                                self.selected[i]=absolute;removed_now.update(absolute)
                                self.records.append(dict(stage=i,group=stage['group'],layer=_layer,step=self.step,
                                    applies_to_layers=list(range(_layer,len(self.blocks))),drop_budget=budget,
                                    candidate_count=len(pool),requested_fraction=stage['fraction'],
                                    dropped_positions=list(absolute),probe_positions=self.probes.detach().cpu().tolist(),
                                    removed_attention_mass=float(scores.index_select(0,chosen_local).sum()),
                                    dropped_by_group={g:len(set(absolute).intersection(self.prepared.group_positions(g)))
                                                      for g in ('acoustic','semantic','prompt')}))
                        with self.scope('prune.compact_rows'):
                            keep=self.keep_rows(positions,tuple(sorted(removed_now)))
                            if keep is not None:
                                x=x.index_select(1,keep);positions=positions.index_select(0,keep)
                                q,k,v=q.index_select(-2,keep),k.index_select(-2,keep),v.index_select(-2,keep)
                    rows=x.shape[1]
                    attended=_block._scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False)
                    attended=attended.transpose(1,2).contiguous().view(B,rows,C)
                    x=x+_block.dropout(_block.attn_out(attended))
                    norm=_block.ff_norm(x)
                    x=x+_block.dropout(_block.ff_out(_block.act(_block.ff_proj(norm))*_block.up_proj(norm)))
                    self.positions=positions
                    event['layer_qkv_rows'].append(qkv_rows)
                    event['layer_computed_rows'].append(rows)
                    event['layer_kv_columns'].append(rows)
                    if _layer==len(self.blocks)-1 and rows!=self.length:
                        with self.scope('prune.restore_output'):
                            full=x.new_zeros((B,self.length,C));full.index_copy_(1,positions,x);x=full
                    return x,None
                block.forward=forward
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*args):
        for block,existed,original in reversed(self.originals):
            if existed:block.forward=original
            else:delattr(block,'forward')
        self.originals.clear()

    def accounting(self):
        if any(len(e['layer_computed_rows'])!=len(self.blocks) for e in self.events):
            raise RuntimeError('Incomplete pruning forward')
        dense=sum(e['total_rows']*len(self.blocks) for e in self.events)
        used=sum(sum(e['layer_computed_rows']) for e in self.events)
        return dict(forwards=len(self.events),computed_token_layers=used,dense_token_layers_at_same_forward_count=dense,
            skipped_token_update_fraction=1-used/max(1,dense),
            qkv_token_layers=sum(sum(e['layer_qkv_rows']) for e in self.events),
            computed_rows_by_layer=[sum(e['layer_computed_rows'][l] for e in self.events) for l in range(len(self.blocks))],
            kv_columns_by_layer=[sum(e['layer_kv_columns'][l] for e in self.events) for l in range(len(self.blocks))],
            qk_query_key_pairs_by_layer=[sum(e['layer_computed_rows'][l]*e['layer_kv_columns'][l] for e in self.events) for l in range(len(self.blocks))],
            effective_schedules=[],pruning=dict(specification=self.spec,selections=self.records,
                activated_layers=len({l for r in self.records for l in r['applies_to_layers']})),
            note='No temporal KV/hidden banks. QKV count includes full selection-boundary projections; row count describes MLP/attention queries. '
                 'Rows remain compact between layers; original RoPE coordinates retained. Final output restored once for unchanged full vocabulary head.')
