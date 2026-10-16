"""CPU-only scheduling; positions and inclusive window bounds are response-relative.

No confidence or entropy gating: only freshly predicted positions may be selected.
The caller keeps prompt rows awake; sleepers can include previously filled rows.
"""


def merge_windows(windows):
    """Union overlapping or directly touching inclusive intervals."""
    merged = []
    for left, right in sorted(windows):
        if left > right:
            raise ValueError('Reversed window')
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


class WindowSchedule:
    """Commit-centered windows persist, expand, merge, and retire when exhausted.

    Every refresh_every forwards (including step zero), all response rows wake.
    If no window contains a MASK, an extra global refresh guarantees progress.
    Refresh cadence is measured in generation forwards, not committed token count.
    """

    def __init__(self, length, mode, *, radius=4, refresh_every=8, **unused):
        if length < 1 or radius < 0 or refresh_every < 1:
            raise ValueError('Require length>=1, radius>=0, refresh_every>=1')
        if mode not in ('baseline', 'simulated_sleep'):
            raise ValueError('Unknown scheduling mode')
        self.length, self.radius, self.refresh_every = length, radius, refresh_every
        self.baseline = mode == 'baseline'
        self.windows = []
        self.last_commits = []
        self.stats = dict(global_refreshes=0, periodic_refreshes=0, empty_window_refreshes=0,
                          windows_opened=0, new_region_seeds=0, windows_merged=0, windows_retired=0,
                          window_count_sum=0, max_window_count=0,
                          sleeping_filled_position_forwards=0)
        self.fallback = False
        self.event = {}

    def prepare(self, step, state, mask_id):
        if len(state) != self.length:
            raise ValueError('Wrong response length')
        masked = {p for p, token in enumerate(state) if token == mask_id}
        retained = [(a,b) for a,b in self.windows if any(a <= p <= b for p in masked)]
        retired = len(self.windows)-len(retained)
        self.windows = retained
        periodic = step % self.refresh_every == 0
        self.fallback = not self.baseline and not periodic and not retained and bool(masked)
        full = self.baseline or periodic or self.fallback
        # Never leave a newly revealed token represented by MASK K/V just because
        # its window exhausted. Refresh its row at least once after commitment.
        sleepers = [] if full else [p for p in range(self.length)
                                   if p not in self.last_commits and
                                   not any(a <= p <= b for a,b in retained)]
        self.stats['windows_retired'] += retired
        if not self.baseline:
            self.stats['global_refreshes'] += int(full)
            self.stats['periodic_refreshes'] += int(periodic)
            self.stats['empty_window_refreshes'] += int(self.fallback)
        self.stats['window_count_sum'] += len(retained)
        self.stats['max_window_count'] = max(self.stats['max_window_count'],len(retained))
        self.stats['sleeping_filled_position_forwards'] += sum(p not in masked for p in sleepers)
        self.event = {'windows_before': [list(w) for w in retained], 'global_refresh':full,
                      'refresh_reason': 'baseline' if self.baseline else 'periodic' if periodic
                                        else 'empty_windows' if self.fallback else None,
                      'retired_count':retired, 'opened_seeds':[], 'merge_count':0,
                      'fresh_previous_commits':list(self.last_commits)}
        return sleepers

    def finish(self, step, rows, committed):
        if len(set(committed)) != len(committed) or any(p < 0 or p >= self.length for p in committed):
            raise ValueError('Invalid commitments')
        if self.baseline:
            return []
        self.last_commits = list(committed)
        # Determine new-region seeds against PRE-batch windows. Same-batch seeds
        # may immediately merge; report both births and resulting interval union.
        opened = [p for p in committed if not any(a <= p <= b for a,b in self.windows)]
        additions = [(max(0,p-self.radius),min(self.length-1,p+self.radius)) for p in committed]
        before_count = len(self.windows)
        # Absorb extensions touching existing windows; count many-to-one OLD
        # window mergers separately from overlapping seeds in the same batch.
        updated = merge_windows(self.windows+additions)
        mergers = sum(max(0,sum(a <= left and right <= b for left,right in self.windows)-1)
                      for a,b in updated)
        self.windows = updated
        old_windows = self.event['windows_before']
        births = sum(not any(a <= left and right <= b for left,right in old_windows)
                     for a,b in updated)
        self.stats['windows_opened'] += births
        self.stats['new_region_seeds'] += len(opened)
        self.stats['windows_merged'] += mergers
        self.stats['max_window_count'] = max(self.stats['max_window_count'],len(updated))
        self.event.update(windows_after=[list(w) for w in updated], opened_seeds=opened,
                          merge_count=mergers, window_count_before=before_count,
                          best_confidence_by_window=[max((r['confidence'] for r in rows
                              if a<=r['position']<=b),default=None) for a,b in old_windows],
                          best_outside_window_confidence=max((r['confidence'] for r in rows
                              if not any(a<=r['position']<=b for a,b in old_windows)),default=None))
        return []

    def snapshot(self):
        return dict(self.event)
