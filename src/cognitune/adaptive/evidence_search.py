"""Opt-in evidence policy: measured coverage, supported replacement and regions.

No comparator observations or comparator best configurations enter this policy.
"""
from __future__ import annotations


from .stable_search import StableSearch


class EvidenceSearch(StableSearch):
    CORE = ('work_mem', 'shared_buffers', 'max_parallel_workers_per_gather', 'jit')
    RELEVANT = CORE + ('effective_cache_size', 'effective_io_concurrency',
                       'random_page_cost', 'seq_page_cost')

    def __init__(self, space, ownership, **kwargs):
        super().__init__(space, ownership, episode=12, minimum_tests=4, max_swaps=1, **kwargs)
        for node in self.state['nodes'].values():
            node.setdefault('isolated_tests', 0)
            node.setdefault('representative_values', [])

    def priority(self, name):
        return 3. if name in self.CORE else 1.5 if name in self.RELEVANT else 1.

    def select(self, proposed, potentials, ranked_groups, trial, protected, limit, delta):
        if not self.state['active']:
            return super().select(proposed, potentials, ranked_groups, trial, protected, limit, delta)
        active = set(self.state['active'])
        report = {'review':False, 'reason':'retain_until_measured', 'swaps':[]}
        if not self.ablation.dynamic_knobs or trial-self.state['last_review'] < self.episode:
            return active, report
        self.state['last_review'] = trial
        report.update(review=True, reason='measured_evidence_review')
        nodes = self.state['nodes']
        linked = set()
        for group in ranked_groups:
            info=potentials.get('+'.join(group),{})
            if len(group)>1 and set(group)<=active and info.get('p_improve',0)>=.7 and info.get('expected_gain',0)>delta:
                linked.update(group)
        outgoing = [n for n in active if n not in protected
                    and n not in linked
                    and trial-nodes[n]['entered'] >= self.episode
                    and nodes[n]['tests'] >= 4 and nodes[n]['isolated_tests'] >= 2
                    and nodes[n]['fresh_tests'] >= 2
                    and len(nodes[n]['tested_values']) >= self.distinct_tests_required(n)
                    and potentials[n]['state'] == 'pause'
                    and (nodes[n]['last_gain'] is None or trial-nodes[n]['last_gain'] >= self.episode)]
        # Uncertainty/age by itself is not a reason to evict a tested direction.
        incoming = [n for n in self.space.by_name if n not in active
                    and nodes[n]['lifetime_tests'] >= 2
                    and potentials[n]['p_improve'] >= .7]
        for old in sorted(outgoing, key=lambda n:potentials[n]['expected_gain']):
            for new in sorted(incoming, key=lambda n:-potentials[n]['expected_gain']):
                k = self.space.by_name[new]
                if k.condition and k.condition.parent not in active:
                    continue
                if potentials[new]['interval'][0] <= potentials[old]['interval'][1] + delta:
                    continue
                candidate = (active-{old}) | {new}
                if any(any(e in self.ownership[n] for n in active)
                       and not any(e in self.ownership[n] for n in candidate)
                       for e in set(e for owners in self.ownership.values() for e in owners)):
                    continue
                nodes[new].update(entered=trial, tests=0, fresh_tests=0,
                                  isolated_tests=0, tested_values=[], window=[])
                self.state['active'] = sorted(candidate)
                report['swaps'].append({'out':[old], 'in':[new], 'reason':'measured_saturation_and_supported_alternative',
                                       'exit_tests':{old:nodes[old]['tests']},
                                       'exit_isolated_tests':{old:nodes[old]['isolated_tests']}})
                return candidate, report
        return active, report

    def observe(self, anchor, config, utility, anchor_utility, trial, delta, unit, *, source, reference=False):
        if reference:
            return []
        changed = [n for n in set(anchor)&set(config) if anchor[n] != config[n]]
        if len(changed) > 4:
            return []
        updates = []
        for name in changed:
            node = self.state['nodes'][name]
            coordinate = unit(self.space.by_name[name], config[name])
            if not any(abs(coordinate-v)<.005 for v in node['representative_values']):
                node['representative_values'].append(coordinate)
            if name not in self.state['active']:
                node['lifetime_tests'] += 1  # actual re-entry trials, not GP probes
                continue
            numeric = self.space.by_name[name].kind in {'integer','float'}
            distance = coordinate-unit(self.space.by_name[name], anchor[name])
            if numeric and abs(distance)<min(.01, node['width']*.1):
                continue
            for key in ['tests','lifetime_tests','fresh_tests']:
                node[key] += 1
            node['isolated_tests'] += int(len(changed)==1)
            if not any(abs(coordinate-v)<.005 for v in node['tested_values']):
                node['tested_values'].append(coordinate)
            gain = utility-anchor_utility
            threshold = delta
            if gain > threshold:
                node['last_gain'] = trial
            if not numeric or not self.ablation.dynamic_regions:
                continue
            node['window'].append(dict(position=distance/node['width'], gain=gain, trial=trial))
            node['window'] = node['window'][-8:]
            window = node['window']
            if len(window)<4:
                continue
            width = node['width']
            positive = [x for x in window if x['gain']>threshold]
            boundary_bad = [x for x in window if abs(x['position'])>=.7 and x['gain'] < -threshold]
            covered = min(x['position'] for x in window)<-.25 and max(x['position'] for x in window)>.25
            reason = None
            if any(abs(x['position'])>=.7 for x in positive):
                new, reason = min(.5, width*1.5), 'supported_boundary_expand'
            elif len(positive)>=2 and len(boundary_bad)>=2:
                new, reason = max(.05, width*.8), 'supported_interior_over_boundary_refine'
            elif covered and trial-(node['last_gain'] if node['last_gain'] is not None else node['entered'])>=12:
                new, reason = min(.5, width*1.5), 'unresolved_plateau_expand'
            if reason and abs(new-width)>1e-12:
                update = dict(knob=name, before=width, after=new, reason=reason,
                              threshold=threshold, evidence=list(window), trial=trial)
                node['width'] = new
                node['window'] = []
                node['range_changes'].append(update)
                updates.append(update)
        return updates

    def coverage(self, anchor, space, seen, unit, value_at, bounded_patch):
        """One legal unseen representative, so EI cannot repeatedly veto it."""
        active = self.state['active']
        def choices_for(k):
            if k.kind in {'integer','float'}:
                return [k.default,k.lower,k.upper]+[value_at(k,u) for u in [.25,.5,.75]]
            return [False,True] if k.kind=='bool' else list(k.choices)
        def representative_debt(name):
            k=space.by_name[name]
            known=self.state['nodes'][name]['representative_values']+[unit(k,anchor.get(name,k.default))]
            return any(not any(abs(unit(k,value)-v)<.005 for v in known) for value in choices_for(k))
        ordered = sorted(active, key=lambda n:(
            0 if n in self.CORE and representative_debt(n) else 1,
            (self.state['nodes'][n]['isolated_tests']+.5)/self.priority(n), n))
        for name in ordered:
            k = space.by_name[name]
            node = self.state['nodes'][name]
            choices = choices_for(k)
            for value in choices:
                coordinate = unit(k,value)
                if any(abs(coordinate-v)<.005 for v in node['representative_values']):
                    continue
                cfg = bounded_patch(space,anchor,{name:value},space.by_name)
                if cfg is not None and cfg!=anchor and space.config_key(cfg) not in seen:
                    return [cfg], [name]
        return [], []
