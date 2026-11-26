"""November paper policy: normalized allocation and expert/V-ranked regrouping.

Kept separate from the measured coverage-v1 and the proposed revised-v2.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from .revised_coverage_search import RevisedCoverageSearch
from .coverage_search import CoverageSearch, CoverageSettings
from .coverage_proposals import CoverageProposer
from .epoch_search import joint_gains


@dataclass(frozen=True)
class PaperCoverageSettings(CoverageSettings):
    coverage_weight: float = .2
    regroup_after: int = 3
    macro_min_changes: int = 3

    def __post_init__(self):
        if self.coverage_weight <= 0 or not np.isfinite(self.coverage_weight):
            raise ValueError('coverage_weight must be finite and positive')
        if self.regroup_after < 1:
            raise ValueError('regroup_after must be positive')


class PaperCoverageSearch(RevisedCoverageSearch):
    allocation_policy = 'normalized_mean_NEI_times_coverage_fixed_lambda'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('settings', PaperCoverageSettings())
        super().__init__(*args, **kwargs)

    def allocation_context(self, means):
        return {'nei_scale': max(means.values(), default=0.),
                'coverage_weight': self.settings.coverage_weight}

    def candidate_pool(self, *args, **kwargs):
        configs, sources, report = super().candidate_pool(*args, **kwargs)
        scale = max(report['mean_nei'].values(), default=0.)
        report['normalized_mean_nei'] = {k:v/scale if scale>0 else 0.
                                       for k,v in report['mean_nei'].items()}
        report['fixed_lambda'] = self.settings.coverage_weight
        return configs, sources, report

    def local_record(self, key, mode):
        # Count actual executions of a combination; numerical region changes
        # and a changed direction ID must not erase its execution evidence.
        signature = repr((mode['expert'], tuple(sorted(mode['knobs']))))
        records = self.state.setdefault('combination_evidence', {})
        record = records.setdefault(signature, {'signature': signature,
            'executions': 0, 'stale': 0, 'last_review_stale': -1,
            'reviewed_executions': 0})
        self.state['local'][key] = record
        return record

    def observe_result(self, plan, result):
        # Deduplicated configurations can have several source direction IDs;
        # the same expert/combination is counted once, not once per ID.
        unique = {}; original = plan.get('sources', [])
        for key in original:
            mode = self.state['modes'].get(key)
            if mode:
                unique.setdefault((mode['expert'], tuple(sorted(mode['knobs']))), key)
        adjusted = dict(plan, sources=list(unique.values()))
        CoverageSearch.observe_result(self, adjusted, result)
        counts = self.state.setdefault('adaptive_knob_change_counts', {})
        for name in self.state['executions'][-1]['changed_knobs']:
            counts[name] = counts.get(name, 0) + 1

    def regroup_requests(self, gp, anchor, rng, seen):
        due = {}
        for key, mode in sorted(self.state['modes'].items()):
            record = self.local_record(key, mode)
            if record['stale'] < self.settings.regroup_after:
                continue
            if record['executions'] - record.get('reviewed_executions', 0) < self.settings.regroup_after:
                continue
            inside = self.sample(mode, anchor, rng, self.settings.probes, seen)
            outside = self.sample(mode, anchor, rng, self.settings.probes, seen, outside=True)
            gains, _, _ = joint_gains(gp, anchor, [inside, outside], rng,
                                     self.settings.posterior_samples, self.settings.delta)
            due[key] = {'expert': mode['expert'], 'knobs': mode['knobs'],
                'local_executions': record['executions'], 'stale_local_executions': record['stale'],
                'inside_opportunity_probability': float(np.mean(gains[0] > 0)),
                'outside_opportunity_probability': float(np.mean(gains[1] > 0)),
                'inside_expected_gain': float(gains[0].mean()),
                'outside_expected_gain': float(gains[1].mean()),
                'inside_probe_count': len(inside), 'outside_probe_count': len(outside),
                'reason': 'three_local_stale_expert_review'}
        self.state['regroup_requests'] = due
        return {row['expert'] for row in due.values()}

    def choose_group_indices(self, gains, choices):
        values = [float(g.mean()) for g in gains]
        return sorted(range(len(choices)), key=lambda i: (-values[i], not choices[i][0], i))[:2]

    def micro(self, gp, anchor, rng, seen):
        # Regions use Q; combinations compete by equal-size V as specified in
        # the latest paper. These are intentionally different decisions.
        requested = copy.deepcopy(self.state['regroup_requests'])
        proposals = self.state['proposals']
        result = {}; reports = {}
        for key, mode in sorted(self.state['modes'].items()):
            result[key], reports[key] = self.micro_mode(gp, anchor, mode, rng, seen)
        experts = {r['expert'] for r in requested.values()}
        for expert in sorted(experts):
            answer = proposals.get(expert, {})
            replace = any(answer.get('regroup_reviews', {}).get(key, {}).get('action') == 'replace'
                          for key, r in requested.items() if r['expert'] == expert)
            if not replace:
                continue
            existing = [(key, m) for key, m in result.items() if m['expert'] == expert]
            choices = [(True, key, m) for key, m in existing]
            groups = {tuple(sorted(m['knobs'])) for _, m in existing}
            for g in answer.get('groups', [])[:2]:
                names = self.closure(g['knobs']); signature = tuple(sorted(names))
                if not names or not names <= set(self.state['active']) or len(names)>self.settings.max_group_knobs:
                    continue
                if signature in groups:
                    continue
                groups.add(signature)
                choices.append((False, '', self.make_mode(expert, names, answer, anchor, 'proposal')))
            pools = [self.sample(m, anchor, rng, self.settings.probes, seen) for _, _, m in choices]
            count = min(map(len, pools), default=0)
            if not count:
                continue
            gains, _, _ = joint_gains(gp, anchor, [p[:count] for p in pools], rng,
                                      self.settings.posterior_samples, self.settings.delta)
            values = [float(g.mean()) for g in gains]
            order = self.choose_group_indices(gains, choices)
            ranking = [{'knobs':m['knobs'], 'existing':old, 'value':values[i],
                        'selected':i in order, 'count':count} for i,(old,_,m) in enumerate(choices)]
            for key, _ in existing:
                del result[key]
            for slot, i in enumerate(order):
                old, previous_key, mode = choices[i]
                key = f'{expert}:{slot}'; mode = copy.deepcopy(mode);mode['id'] = key
                result[key] = mode
            for key, request in requested.items():
                if request['expert'] == expert:
                    reports.setdefault(key, {})['regroup_ranking'] = ranking
                    if any(not choices[i][0] for i in order):
                        reports[key]['group_change'] = {'selected_groups':[choices[i][2]['knobs'] for i in order]}
        self.state['modes'] = self.ensure_coverage(result,set(self.state['active']),proposals,anchor)
        for key, request in requested.items():
            answer = proposals.get(request['expert'], {})
            reports.setdefault(key, {})['expert_regroup_review'] = copy.deepcopy(
                answer.get('regroup_reviews', {}).get(key, {}))
            # Mark the *old combination* as reviewed even if it was replaced.
            signature = repr((request['expert'], tuple(sorted(request['knobs']))))
            record = self.state['combination_evidence'][signature]
            record.update(stale=0, reviewed_executions=record['executions'])
        self.state['regroup_requests'] = {}
        return reports

    def macro(self, gp, anchor, rng, seen, adaptive_completed, answers):
        original = self.state['ever_changed']
        counts = self.state.get('adaptive_knob_change_counts', {})
        self.state['ever_changed'] = [n for n in original if counts.get(n,0)>=self.settings.macro_min_changes]
        try:
            report = super().macro(gp,anchor,rng,seen,adaptive_completed,answers)
            report['adaptive_knob_change_counts'] = dict(counts)
            report['minimum_actual_changes'] = self.settings.macro_min_changes
            return report
        finally:
            self.state['ever_changed'] = original


class PaperCoverageProposer(CoverageProposer):
    system = CoverageProposer.system + (
        ' For phase regroup, explicitly review EVERY requested direction owned by you in '
        'regroup_evidence. Return regroup_reviews:{direction_id:{action:"continue" or '
        '"replace",reason:string}}. Base the decision on full history, actual local '
        'executions, inside/outside GP opportunities and sampled coverage. Low observed '
        'gain alone is not exhaustion. For continue, preserve that combination. For '
        'replace, propose at most two alternative active-only combinations in groups. '
        'The controller compares existing and proposed groups using equal-size '
        'expected best improvement V and keeps at most two; region revisions use Q.'
    )

    def __call__(self, expert, context):
        return super().__call__(expert, dict(context, expert=expert))

    @staticmethod
    def validate(answer, context):
        CoverageProposer.validate(answer, context)
        if context['phase'] != 'regroup':
            return
        requested = {key for key, row in context.get('regroup_evidence', {}).items()
                     if row['expert'] == context['expert']}
        reviews = answer.get('regroup_reviews', {})
        if not isinstance(reviews, dict) or not requested <= set(reviews):
            raise ValueError('regroup_reviews must review every requested owned direction')
        for key in requested:
            row = reviews[key]
            if not isinstance(row, dict) or row.get('action') not in {'continue', 'replace'} or not row.get('reason'):
                raise ValueError('regroup review requires continue/replace and reason')
        if any(reviews[key]['action'] == 'replace' for key in requested) and not answer.get('groups'):
            raise ValueError('replace decision requires an alternative group proposal')
