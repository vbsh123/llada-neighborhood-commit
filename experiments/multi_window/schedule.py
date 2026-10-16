"""Persistent MASK-gap regions with padded computation windows; CPU scheduling."""


def merge_windows(windows):
    """Union overlapping or directly touching inclusive computation intervals."""
    merged = []
    for left, right in sorted(windows):
        if left > right:
            raise ValueError('Reversed window')
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def region_bounds(filled, gap):
    """Two filled groups are separate iff >=gap consecutive MASKs lie between.

    Bounds include every intervening MASK when the gap is smaller than gap.
    For gap=4, filled positions 2 and 6 share a region; 2 and 7 do not.
    """
    if gap < 1:
        raise ValueError('Require region_gap>=1')
    result = []
    for p in sorted(set(filled)):
        if result and p-result[-1][1]-1 < gap:
            result[-1] = (result[-1][0], p)
        else:
            result.append((p,p))
    return result


def component_changes(before, after):
    """New components and mergers of old components; monotonic filling only."""
    parent_counts = [sum(a <= left and right <= b for left,right in before)
                     for a,b in after]
    return sum(n == 0 for n in parent_counts), sum(max(0,n-1) for n in parent_counts)


class WindowSchedule:
    """Every historical region stays fresh, including filled tokens and interior.

    Padded windows are recomputed around ALL regions, regardless of recent activity.
    Periodic global forwards discover distant positions. No retirement or timeout.
    Logical region mergers depend on MASK gaps, not overlapping window padding.
    """

    def __init__(self, length, mode, *, radius=4, region_gap=4, refresh_every=8, **unused):
        if length < 1 or radius < 0 or region_gap < 1 or refresh_every < 1:
            raise ValueError('Require length>=1, radius>=0, region_gap>=1, refresh_every>=1')
        if mode not in ('baseline', 'simulated_sleep'):
            raise ValueError('Unknown scheduling mode')
        self.length, self.radius, self.region_gap = length, radius, region_gap
        self.refresh_every = refresh_every
        self.baseline = mode == 'baseline'
        self.windows, self.regions, self.filled = [], [], set()
        self.stats = dict(global_refreshes=0, periodic_refreshes=0, empty_window_refreshes=0,
                          windows_opened=0, windows_merged=0, windows_retired=0,
                          regions_opened=0, regions_merged=0, new_region_seeds=0,
                          window_count_sum=0, max_window_count=0,
                          region_count_sum=0, max_region_count=0,
                          sleeping_filled_position_forwards=0)
        self.fallback = False
        self.event = {}

    def geometry(self, filled):
        regions = region_bounds(filled,self.region_gap)
        windows = merge_windows([(max(0,a-self.radius),min(self.length-1,b+self.radius))
                                 for a,b in regions])
        return regions,windows

    def prepare(self, step, state, mask_id):
        if len(state) != self.length:
            raise ValueError('Wrong response length')
        self.filled = {p for p, token in enumerate(state) if token != mask_id}
        self.regions,self.windows = self.geometry(self.filled)
        covered = {p for a,b in self.windows for p in range(a,b+1)}
        periodic = step % self.refresh_every == 0
        has_candidate = bool(covered-self.filled)
        self.fallback = not self.baseline and not periodic and not has_candidate
        full = self.baseline or periodic or self.fallback
        sleepers = [] if full else [p for p in range(self.length) if p not in covered]
        if not self.baseline:
            self.stats['global_refreshes'] += int(full)
            self.stats['periodic_refreshes'] += int(periodic)
            self.stats['empty_window_refreshes'] += int(self.fallback)
        for name,components in [('window',self.windows),('region',self.regions)]:
            self.stats[f'{name}_count_sum'] += len(components)
            self.stats[f'max_{name}_count'] = max(self.stats[f'max_{name}_count'],len(components))
        # Every filled row is contained in a region and therefore always awake.
        if self.filled.intersection(sleepers):
            raise RuntimeError('Filled response positions must remain awake')
        self.event = {'windows_before':[list(w) for w in self.windows],
                      'regions_before':[list(r) for r in self.regions], 'global_refresh':full,
                      'refresh_reason':'baseline' if self.baseline else 'periodic' if periodic
                                       else 'empty_windows' if self.fallback else None,
                      'retired_count':0,'opened_seeds':[],'merge_count':0}
        return sleepers

    def finish(self, step, rows, committed):
        if len(set(committed)) != len(committed) or any(p < 0 or p >= self.length for p in committed):
            raise ValueError('Invalid commitments')
        if self.baseline:
            return []
        before_regions,before_windows = self.regions,self.windows
        after_regions,after_windows = self.geometry(self.filled.union(committed))
        births,mergers = component_changes(before_regions,after_regions)
        window_births,window_mergers = component_changes(before_windows,after_windows)
        # A new-region seed belongs to a post-batch region with NO old parent.
        # Padding overlap does not determine whether a region is new.
        new_regions = [(a,b) for a,b in after_regions
                       if not any(a<=left and right<=b for left,right in before_regions)]
        seeds = [p for p in committed if any(a<=p<=b for a,b in new_regions)]
        self.stats['regions_opened'] += births
        self.stats['regions_merged'] += mergers
        self.stats['windows_opened'] += window_births
        self.stats['windows_merged'] += window_mergers
        self.stats['new_region_seeds'] += len(seeds)
        self.stats['max_region_count'] = max(self.stats['max_region_count'],len(after_regions))
        self.stats['max_window_count'] = max(self.stats['max_window_count'],len(after_windows))
        self.regions,self.windows = after_regions,after_windows
        self.event.update(windows_after=[list(w) for w in after_windows],
                          regions_after=[list(r) for r in after_regions],opened_seeds=seeds,
                          merge_count=window_mergers,region_merge_count=mergers,
                          window_count_before=len(before_windows),
                          best_confidence_by_window=[max((r['confidence'] for r in rows
                              if a<=r['position']<=b),default=None) for a,b in before_windows],
                          best_outside_window_confidence=max((r['confidence'] for r in rows
                              if not any(a<=r['position']<=b for a,b in before_windows)),default=None))
        return []

    def snapshot(self):
        return dict(self.event)
