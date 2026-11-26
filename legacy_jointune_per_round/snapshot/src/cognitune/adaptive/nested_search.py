"""Opt-in HierTune policy: test value support before replacing coordinates.

All opportunities are conditional on one anchor. Finite probes are not global
upper bounds; only actual, context-matched executions establish coverage.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .decision import allocate, factor
from .stable_search import StableSearch


@dataclass(frozen=True)
class NestedSettings:
    probes: int = 16
    minimum_tests: int = 4
    outside_tests: int = 2
    stale_executions: int = 6
    low_probability: float = .1
    admission_probability: float = .7
    comparison_probability: float = .8
    minimum_width: float = .05
    maximum_width: float = .5


def group_key(group):
    return '+'.join(sorted(group))


def strict_patch(space, anchor, patch, allowed, **kwargs):
    """Reject resource/activation repair that changes undeclared coordinates."""
    if not isinstance(patch, dict) or not patch or not set(patch) <= set(allowed):
        return None
    try:
        for name, value in patch.items():
            knob = space.by_name[name]
            if knob.kind in {'integer', 'float'}:
                if isinstance(value, (bool, str)) or not np.isfinite(value):
                    return None
                if not knob.lower <= value <= knob.upper:
                    return None
                if knob.kind == 'integer' and float(value) != int(value):
                    return None
            elif knob.kind == 'bool':
                if not isinstance(value, (bool, np.bool_)):
                    return None
            elif value not in knob.choices:
                return None
        result = space.repair({**anchor, **patch})
        changed = {n for n in set(anchor) | set(result) if anchor.get(n) != result.get(n)}
        if not changed <= set(allowed) or not space.validate(result):
            return None
        return result
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
        return None


def inside_region(space, config, group, regions, unit):
    for name in group:
        if name not in config:
            return False
        knob, region = space.by_name[name], regions[name]
        if knob.kind in {'integer', 'float'}:
            if abs(unit(knob, config[name]) - region['center']) > region['halfwidth'] + 1e-12:
                return False
        elif config[name] not in region['values']:
            return False
    return True


def outside_exists(space, group, regions):
    for name in group:
        k, r = space.by_name[name], regions[name]
        if k.kind in {'integer', 'float'}:
            if r['center'] - r['halfwidth'] > 0 or r['center'] + r['halfwidth'] < 1:
                return True
        else:
            choices = [False, True] if k.kind == 'bool' else list(k.choices)
            if set(choices) - set(r['values']):
                return True
    return False


def probe_domains(space, anchor, group, regions, rng, count, unit, value_at):
    """Equal target sizes, unique legal points; never pad a finite domain."""
    pools = {'inside': [], 'outside': []}
    applicable = outside_exists(space, group, regions)
    for scope in pools:
        if scope == 'outside' and not applicable:
            continue
        seen = set()
        for _ in range(max(128, count * 64)):
            if len(pools[scope]) >= count:
                break
            patch = {}
            for name in group:
                k, r = space.by_name[name], regions[name]
                if k.kind in {'integer', 'float'}:
                    lo, hi = max(0, r['center'] - r['halfwidth']), min(1, r['center'] + r['halfwidth'])
                    if scope == 'inside':
                        z = rng.uniform(lo, hi)
                    else:
                        # Half full-domain, half immediately outside a boundary.
                        z = rng.random()
                        if rng.random() < .5:
                            intervals = [(max(0, lo-r['halfwidth']), lo)] if lo > 0 else []
                            if hi < 1:
                                intervals.append((hi, min(1, hi+r['halfwidth'])))
                            if intervals:
                                left, right = intervals[int(rng.integers(len(intervals)))]
                                z = rng.uniform(left, right)
                    patch[name] = value_at(k, z)
                else:
                    values = ([False, True] if k.kind == 'bool' else list(k.choices)) if scope == 'outside' else r['values']
                    patch[name] = values[int(rng.integers(len(values)))]
            cfg = strict_patch(space, anchor, patch, group)
            if cfg is None or inside_region(space, cfg, group, regions, unit) != (scope == 'inside'):
                continue
            key = space.config_key(cfg)
            if key not in seen:
                seen.add(key)
                pools[scope].append(cfg)
    pools['outside_applicable'] = applicable
    return pools


def gain_summary(gains, count, delta):
    if count == 0:
        return {'p_improve': .5, 'expected_gain': 0., 'interval': [None, None],
                'state': 'verify', 'count': 0, 'available': False}
    p = float(np.mean(gains > delta))
    lo, hi = map(float, np.quantile(gains, [.1, .9]))
    return {'p_improve': p, 'expected_gain': float(np.mean(np.maximum(gains-delta, 0))),
            'interval': [lo, hi], 'state': 'pause' if p <= .1 and hi-lo <= 2*delta else 'verify',
            'count': count, 'available': True}


def evaluate_domains(gp, anchor, pools, rng, samples, delta):
    """One joint latent draw bank; full is a union, not an independent model."""
    configs = [anchor] + pools['inside'] + pools['outside']
    mean, covariance, _ = gp.joint_posterior(configs)
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError('non-finite nested posterior')
    draws = mean + rng.standard_normal((samples, len(configs))) @ factor(covariance).T
    n = len(pools['inside'])
    result = {}
    for label, a, b in [('inside', 1, 1+n), ('outside', 1+n, len(configs)), ('full', 1, len(configs))]:
        gain = np.max(draws[:, a:b] - draws[:, :1], axis=1) if b > a else np.zeros(samples)
        result[label] = gain_summary(gain, b-a, delta)
    result['outside_applicable'] = pools['outside_applicable']
    return result


def compare_pools(gp, anchor, old, new, rng, samples, delta):
    """Same count and common posterior for an admission decision."""
    n = min(len(old), len(new))
    if not n:
        return {'available': False, 'count_each': 0, 'p_advantage': 0.}
    configs = [anchor] + old[:n] + new[:n]
    mean, covariance, _ = gp.joint_posterior(configs)
    draws = np.asarray(mean) + rng.standard_normal((samples, len(configs))) @ factor(np.asarray(covariance)).T
    difference = draws[:, 1+n:].max(1) - draws[:, 1:1+n].max(1)
    return {'available': True, 'count_each': n, 'p_advantage': float(np.mean(difference > delta)),
            'gain_difference_interval': list(map(float, np.quantile(difference, [.1, .9])))}


class NestedSearch(StableSearch):
    """Region repair -> regroup within A -> validated replacement of A.

    One context-matched execution ledger and one opportunity criterion drive
    both levels. Existing StableSearch/EvidenceSearch protocols are untouched.
    """
    def __init__(self, space, ownership, *, settings=None, **kwargs):
        self.settings = settings or NestedSettings()
        super().__init__(space, ownership, minimum_tests=self.settings.minimum_tests, **kwargs)
        self.state.setdefault('nested_evidence', {})
        self.state.setdefault('mode_regions', {})
        self.state.setdefault('nested_opportunities', {})
        self.state.setdefault('micro_decisions', {})
        self.state.setdefault('last_region_update', {})

    def protocol(self):
        return {'version': 'nested-prior-v1', **asdict(self.settings),
                'macro_interval': self.episode, 'max_new_knobs': self.max_swaps,
                'group_external_coordinates': 'strictly fixed to anchor',
                'stagnation_unit': 'successful executions of this group in the same outside context'}

    def _ledger(self, group, anchor):
        key = group_key(group)
        context = self.space.config_key({n: v for n, v in anchor.items() if n not in group})
        node = self.state['nested_evidence'].setdefault(key, {'context': context, 'rows': []})
        if node['context'] != context:
            # Keep observations in the GP/trace, never reuse stale local coverage.
            node = {'context': context, 'rows': []}
            self.state['nested_evidence'][key] = node
        return node

    def _regions(self, group, anchor, regions, unit):
        key = group_key(group)
        if key in self.state['mode_regions']:
            return self.state['mode_regions'][key]
        result = {}
        for name in group:
            k = self.space.by_name[name]
            opinions = [regions[e][name] for e in sorted(regions) if name in regions[e]]
            if k.kind in {'integer', 'float'}:
                result[name] = {'center': float(np.mean([r['center'] for r in opinions])) if opinions else unit(k, anchor.get(name, k.default)),
                                'halfwidth': float(np.clip(self.state['nodes'][name]['width'], self.settings.minimum_width, self.settings.maximum_width))}
            else:
                values = list(dict.fromkeys(v for r in opinions for v in r.get('values', [])))
                result[name] = {'values': values or [anchor.get(name, k.default)]}
        return result

    def assess(self, gp, anchor, regions, groups, rng, unit, value_at, samples, delta, trial):
        self._gp, self._anchor, self._rng = gp, dict(anchor), rng
        self._unit, self._value_at, self._samples, self._delta = unit, value_at, samples, delta
        self._regions_input = regions
        self._probe_sets, self._probe_means = {}, {}
        due = not self.state['active'] or trial-self.state['last_review'] >= self.episode
        self.state['micro_decisions'] = {}
        wanted = list(dict.fromkeys(tuple(sorted(g)) for g in groups))
        potentials = copy.deepcopy(self.state['nested_opportunities'])
        for group in sorted(wanted):
            key = group_key(group)
            if due or len(group) > 1 or key not in potentials or any(set(group) == set(h['knobs']) for h in self.state['hypotheses'].values()):
                potentials[key] = self._assess_group(group)
        self.state['nested_opportunities'] = copy.deepcopy(potentials)
        # The optimizer also uses these pools for real, budgeted re-entry trials.
        configs, slices = [anchor], []
        for group in wanted:
            start = len(configs)
            pools = self._probe_sets.get(group_key(group), {})
            configs.extend(pools.get('outside', []) + pools.get('inside', []))
            slices.append((start, len(configs)))
        return potentials, configs, slices

    def _assess_group(self, group):
        self._ledger(group, self._anchor)
        regions = self._regions(group, self._anchor, self._regions_input, self._unit)
        pools = probe_domains(self.space, self._anchor, group, regions, self._rng,
                              self.settings.probes, self._unit, self._value_at)
        evidence = evaluate_domains(self._gp, self._anchor, pools, self._rng, self._samples, self._delta)
        self._probe_sets[group_key(group)] = pools
        # Broad score is conditional finite-probe opportunity, never exhaustion proof.
        return {**evidence['full'], 'nested': evidence, 'regions': copy.deepcopy(regions)}

    def _low(self, summary, delta):
        return (summary['available'] and summary['p_improve'] <= self.settings.low_probability
                and summary['interval'][1]-summary['interval'][0] <= 2*delta)

    def local_status(self, group, opportunity, anchor, unit, delta):
        rows = self._ledger(group, anchor)['rows']
        regions = opportunity['regions']; nested = opportunity['nested']
        outside = [r for r in rows if not inside_region(self.space, r['config'], group, regions, unit)]
        stale = (len(rows) >= self.settings.stale_executions
                 and all(r['gain'] <= delta for r in rows[-self.settings.stale_executions:]))
        covered = (len(rows) >= self.settings.minimum_tests and
                   all(any(r['config'].get(n) != r.get('anchor', anchor).get(n) for r in rows) for n in group))
        low_in = self._low(nested['inside'], delta)
        outer_verified = (not nested['outside_applicable'] or len(outside) >= self.settings.outside_tests)
        low_out = not nested['outside_applicable'] or self._low(nested['outside'], delta)
        if low_in and nested['outside_applicable'] and not low_out:
            decision = 'repair_region' if any(r['gain'] > delta for r in outside) else 'verify_outside'
        elif covered and outer_verified and stale and low_in and low_out:
            decision = 'regroup'
        else:
            decision = 'continue' if covered else 'verify_coverage'
        return {'decision': decision, 'executions': len(rows), 'outside_executions': len(outside),
                'stale_executions': stale, 'outer_verified': outer_verified,
                'local_low': low_in, 'outer_low': low_out}

    def select(self, proposed, potentials, ranked_groups, trial, protected, limit, delta):
        if not self.state['active']:
            return super().select(proposed, potentials, ranked_groups, trial, protected, limit, delta)
        active = set(self.state['active'])
        report = {'review': False, 'reason': 'retain_until_review', 'swaps': []}
        if not self.ablation.dynamic_knobs or trial-self.state['last_review'] < self.episode:
            return active, report
        self.state['last_review'] = trial
        report.update(review=True, reason='nested_opportunity_review')
        linked = set()
        for group in ranked_groups:
            info = potentials[group_key(group)]
            if len(group) > 1 and set(group) <= active and info['available'] and info['p_improve'] >= self.settings.admission_probability:
                linked.update(group)
        outgoing = [tuple(sorted(g)) for g in ranked_groups if set(g) <= active
                    and self._low(potentials[group_key(g)], delta)
                    and self.local_status(g, potentials[group_key(g)], self._anchor, self._unit, delta)['decision'] == 'regroup']
        outgoing.sort(key=lambda g: (potentials[group_key(g)]['expected_gain'], g))
        candidates = []
        for group in ranked_groups:
            closure = set(group)
            for name in tuple(closure):
                k = self.space.by_name[name]
                while k.condition:
                    closure.add(k.condition.parent); k = self.space.by_name[k.condition.parent]
            incoming = closure-active
            if not incoming or len(incoming) > self.max_swaps:
                continue
            if closure != set(group):
                # Parent values must be part of a jointly probed hypothesis.
                continue
            info = potentials[group_key(group)]
            if not info['available'] or info['p_improve'] < self.settings.admission_probability:
                continue
            rows = self._ledger(group, self._anchor)['rows']
            if not any(r['gain'] > delta for r in rows):
                continue  # Proxy opportunity alone never certifies admission.
            candidates.append((info['expected_gain'], tuple(sorted(group)), incoming))
        for _, group, incoming in sorted(candidates, key=lambda x: (-x[0], x[1])):
            for old_group in outgoing:
                old = sorted(set(old_group)-set(group))
                if not old or set(old) & (set(protected) | linked):
                    continue
                candidate = (active-set(old)) | set(group)
                if len(candidate)>limit:
                    continue
                if any(any(e in self.ownership[n] for n in active) and not any(e in self.ownership[n] for n in candidate)
                       for e in self.experts):
                    continue
                if group_key(old_group) not in self._probe_sets:
                    self._assess_group(old_group)
                comparison = compare_pools(self._gp, self._anchor,
                    self._probe_sets[group_key(old_group)]['inside'] + self._probe_sets[group_key(old_group)]['outside'],
                    self._probe_sets[group_key(group)]['inside'] + self._probe_sets[group_key(group)]['outside'],
                    self._rng, self._samples, delta)
                if comparison['p_advantage'] < self.settings.comparison_probability:
                    continue
                self.state['active'] = sorted(candidate)
                for name in incoming:
                    self.state['nodes'][name].update(entered=trial, tests=0, fresh_tests=0, tested_values=[], window=[])
                report['swaps'].append({'out': list(old), 'in': sorted(incoming), 'joint_group': list(group),
                                        'outgoing_evidence_group': list(old_group),
                                        'reason': 'verified_nested_alternative', 'comparison': comparison})
                return candidate, report
        return active, report

    def hypotheses(self, proposals, scopes, active, total, audit_size, confirmed_groups=()):
        previous = copy.deepcopy(self.state['hypotheses'])
        proposed, _ = super().hypotheses(proposals, scopes, active, total, audit_size, confirmed_groups)
        if previous and getattr(self, 'freeze_structure', False):
            proposed = {k: m for k, m in previous.items() if set(m['knobs']) <= set(active)}
        keep = {}
        for key, mode in sorted(previous.items()):
            group = tuple(mode['knobs'])
            if not set(group) <= set(active):
                continue
            info = self.state['nested_opportunities'].get(group_key(group))
            status = self.local_status(group, info, self._anchor, self._unit, self._delta) if info else {'decision': 'verify_coverage'}
            self.state['micro_decisions'][key] = status
            if status['decision'] != 'regroup' or not self.ablation.dynamic_regions or getattr(self, 'freeze_structure', False):
                keep[key] = copy.deepcopy(proposed.get(key, mode))
                self._revise_region(group, info, status)
        capacity = min(12, total//(audit_size+1))
        for key, mode in proposed.items():
            if len(keep) >= capacity:
                break
            for expert in mode['experts']:
                maximum = 8 if self.experts == ('generalist_expert',) else 2
                if sum(expert in m['experts'] for m in keep.values()) >= maximum:
                    break
            else:
                keep.setdefault(key, mode)
        if not keep:
            keep = dict(list(proposed.items())[:capacity])
        keep = {k: keep[k] for k in sorted(keep)}
        quotas, _ = allocate({(k,): self.state['hypothesis_credit'].get(k, 0) for k in keep}
                             if self.ablation.adaptive_allocation else {}, list(keep), total, minimum=audit_size+1)
        self.state['hypotheses'] = copy.deepcopy(keep)
        self.state['hypothesis_quota'] = quotas
        for key, mode in keep.items():
            g = mode['knobs']; gk = group_key(g)
            if gk not in self.state['mode_regions']:
                self.state['mode_regions'][gk] = copy.deepcopy(self._regions(g, self._anchor, self._regions_input, self._unit))
        return keep, quotas

    def _revise_region(self, group, opportunity, status):
        if not opportunity or getattr(self, 'freeze_structure', False):
            return
        key = group_key(group)
        regions = copy.deepcopy(opportunity['regions'])
        self.state['mode_regions'].setdefault(key, copy.deepcopy(regions))
        rows = self._ledger(group, self._anchor)['rows']
        stamp = [r['trial'] for r in rows]
        if (not rows or self.state['last_region_update'].get(key) == stamp
                or not self.ablation.dynamic_regions or getattr(self, 'freeze_structure', False)):
            return
        positive = [r for r in rows if r['gain'] > self._delta]
        if not positive:
            return
        best = max(positive, key=lambda r: r['utility'])
        before = copy.deepcopy(regions)
        for name in group:
            k = self.space.by_name[name]
            if k.kind not in {'integer', 'float'}:
                if best['config'][name] not in regions[name]['values']:
                    regions[name]['values'].append(best['config'][name])
                continue
            r = regions[name]; center = r['center']; width = r['halfwidth']
            positions = [(self._unit(k, row['config'][name])-center)/width for row in rows]
            lower_bad = any(pos <= -.7 and row['gain'] <= self._delta for pos, row in zip(positions, rows))
            upper_bad = any(pos >= .7 and row['gain'] <= self._delta for pos, row in zip(positions, rows))
            boundary_good = any(abs(pos) >= .7 and row['gain'] > self._delta for pos, row in zip(positions, rows))
            if len(rows) >= self.settings.minimum_tests and boundary_good:
                r['halfwidth'] = min(self.settings.maximum_width, width*1.5)
            elif len(rows) >= self.settings.minimum_tests and lower_bad and upper_bad:
                # Both sides really tested: never shrink just because samples are missing.
                r['halfwidth'] = max(self.settings.minimum_width, width*.8)
            # Move to a whole successful joint configuration, not marginal medians.
            r['center'] = float(self._unit(k, best['config'][name]))
        self.state['mode_regions'][key] = regions
        self.state['last_region_update'][key] = stamp
        if regions != before:
            status['region_update'] = {'before': before, 'after': copy.deepcopy(regions),
                                       'evidence_trials': stamp, 'reason': 'executed_local_response'}

    def sample_group(self, mode, regions, anchor, space, seen, rng, n, unit, value_at):
        group = mode['knobs']; effective = self._regions(group, anchor, regions, unit)
        result, local = [], set()
        for _ in range(max(100, n*60)):
            if len(result) >= n:
                break
            patch = {}
            for name in group:
                k, r = space.by_name[name], effective[name]
                if k.kind in {'integer', 'float'}:
                    z = rng.random() if rng.random() < .1 else rng.uniform(max(0, r['center']-r['halfwidth']), min(1, r['center']+r['halfwidth']))
                    patch[name] = value_at(k, z)
                else:
                    choices = [False, True] if k.kind == 'bool' else list(k.choices)
                    values = choices if rng.random() < .1 else r['values']
                    patch[name] = values[int(rng.integers(len(values)))]
            cfg = strict_patch(space, anchor, patch, group)
            if cfg is None:
                continue
            ck = space.config_key(cfg)
            if ck not in seen and ck not in local:
                local.add(ck); result.append(cfg)
        return result

    def observe(self, anchor, config, utility, anchor_utility, trial, delta, unit, *, source, reference=False):
        if reference:
            return []
        changed = {n for n in set(anchor) | set(config) if anchor.get(n) != config.get(n)}
        if not changed or len(changed) > 4:
            return []
        groups = [(n,) for n in self.space.by_name] + [tuple(m['knobs']) for m in self.state['hypotheses'].values()]
        groups += [tuple(k.split('+')) for k in self.state['nested_opportunities']]
        for group in dict.fromkeys(tuple(sorted(g)) for g in groups):
            if not changed <= set(group):
                continue
            ledger = self._ledger(group, anchor)
            ck = self.space.config_key(config)
            if any(r['config_key'] == ck for r in ledger['rows']):
                continue
            ledger['rows'].append({'trial': trial, 'config': dict(config), 'config_key': ck,
                                   'anchor': dict(anchor),
                                   'utility': float(utility), 'gain': float(utility-anchor_utility), 'source': source})
            ledger['rows'] = ledger['rows'][-32:]
        for name in changed:
            node = self.state['nodes'][name]
            for count in ['tests', 'fresh_tests', 'lifetime_tests']:
                node[count] += 1
            z = unit(self.space.by_name[name], config[name])
            if not any(abs(z-v)<.005 for v in node['tested_values']):
                node['tested_values'].append(z)
        return []  # Region decisions occur at the next model/evidence review.

    def coverage(self, anchor, space, seen, unit, value_at, bounded_patch):
        # A scheduled real slot, not all probes, resolves pending outer evidence.
        for key, pools in self._probe_sets.items():
            info = self.state['nested_opportunities'].get(key)
            if not info:
                continue
            group = tuple(key.split('+'))
            if not set(group) <= set(self.state['active']):
                continue
            status = self.local_status(group, info, anchor, unit, self._delta)
            if status['decision'] in {'verify_outside', 'repair_region'} or (status['local_low'] and not status['outer_verified']):
                for cfg in pools['outside']:
                    if space.config_key(cfg) not in seen:
                        return [cfg], list(group)
        for name in sorted(self.state['active'], key=lambda n: (self.state['nodes'][n]['tests'], n)):
            if name not in self._probe_sets:
                self.state['nested_opportunities'][name] = self._assess_group((name,))
            pool = self._probe_sets[name]
            for cfg in pool['inside'] + pool['outside']:
                if cfg != anchor and space.config_key(cfg) not in seen:
                    return [cfg], [name]
        return [], []

    def expert_context(self, expert):
        return {'controller': 'nested-prior-v1', 'scope': 'proposal only; controller accepts revisions',
                'subspaces': [{'knobs': m['knobs'], 'region': self.state['mode_regions'].get(group_key(m['knobs'])),
                               'last_decision': self.state['micro_decisions'].get(k)}
                              for k, m in self.state['hypotheses'].items() if expert in m['experts']]}
