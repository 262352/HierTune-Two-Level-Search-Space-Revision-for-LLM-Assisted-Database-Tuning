"""Preserve audited expert combinations; cover omitted knobs separately.

Two-level revision is unchanged. At most two regular groups per expert compete
by joint V, and auxiliary coverage never rewrites an accepted combination.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .paper_coverage_search import PaperCoverageSearch, PaperCoverageSettings, PaperCoverageProposer


@dataclass(frozen=True)
class FaithfulGroupSettings(PaperCoverageSettings):
    confidence: float = .7


class FaithfulGroupSearch(PaperCoverageSearch):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('settings',FaithfulGroupSettings())
        super().__init__(*args,**kwargs)

    def ensure_coverage(self, modes, active, answers, anchor):
        result = copy.deepcopy(modes)
        for expert in {m['expert'] for m in result.values()}:
            if sum(m['expert']==expert for m in result.values())>2:
                raise ValueError('At most two regular expert groups')
        for mode in result.values():
            if not set(mode['knobs'])<=active or len(mode['knobs'])>self.settings.max_group_knobs:
                raise ValueError('Invalid regular group')
        # A coverage request is a separate sampling task, not a new interaction
        # claim. In particular it cannot append knobs to the selected groups.
        covered={n for mode in result.values() for n in mode['knobs']}
        self.state['coverage_pending']=sorted(active-covered)
        return dict(sorted(result.items()))

    def choose_group_indices(self, gains, choices):
        count=min(2,len(choices));options=list(combinations(range(len(choices)),count))
        def opportunity(indices):
            return float(np.max(np.asarray([gains[i] for i in indices]),axis=0).mean())
        order=max(options,key=lambda ids:(opportunity(ids),sum(choices[i][0] for i in ids),tuple(-i for i in ids)))
        self.state.setdefault('group_bundle_comparisons',[]).append({
            'choices':[{'knobs':m['knobs'],'existing':old,'individual_value':float(gains[i].mean())}
                       for i,(old,_,m) in enumerate(choices)],
            'bundles':[{'indices':list(ids),'joint_value':opportunity(ids)} for ids in options],
            'selected_indices':list(order),'selected_joint_value':opportunity(order)})
        self.state['group_bundle_comparisons']=self.state['group_bundle_comparisons'][-8:]
        return list(order)

    def candidate_pool(self, gp, anchor, rng, seen, total=100, audit_size=4):
        active=set(self.state['active'])
        covered={n for mode in self.state['modes'].values() for n in mode['knobs']}
        missing=sorted(active-covered);reserved=[];reserve_sources={};unavailable=[]
        excluded=set(seen)
        for name in missing:
            names=self.closure([name])
            if not names<=active or len(names)>self.settings.max_group_knobs:
                unavailable.append(name);continue
            expert=max(self.ownership[name],key=self.ownership[name].get)
            mode=self.make_mode(expert,names,{},anchor,'coverage:'+name)
            candidates=self.sample(mode,anchor,rng,1,excluded,full=True)
            if not candidates:
                unavailable.append(name);continue
            cfg=candidates[0];key=self.space.config_key(cfg)
            excluded.add(key);reserved.append(cfg);reserve_sources[key]=['coverage:'+name]
        regular,origins,report=super().candidate_pool(gp,anchor,rng,excluded,total-len(reserved),audit_size)
        configs=regular+reserved;sources={**origins,**reserve_sources}
        report.update(capacity=total,pool_size=len(configs),regular_pool_size=len(regular),
            auxiliary_coverage_count=len(reserved),auxiliary_coverage_knobs=missing,
            auxiliary_coverage_unavailable=unavailable,
            regular_groups={key:list(m['knobs']) for key,m in self.state['modes'].items()},
            group_bundle_comparisons=copy.deepcopy(self.state.get('group_bundle_comparisons',[])),
            candidate_sources=sources,
            active_knob_candidate_exposure={n:sum(c.get(n)!=anchor.get(n) for c in configs) for n in active})
        report['unexposed_active_knobs']=[n for n,c in report['active_knob_candidate_exposure'].items() if not c]
        return configs,sources,report


class FaithfulGroupProposer(PaperCoverageProposer):
    system=PaperCoverageProposer.system+(
        ' Score workload-specific improvement opportunities, not general PostgreSQL importance. '
        'Use the supplied measured transaction mixture, SQL templates, isolation and measurement duration. '
        'A knob affecting background maintenance or complex query planning is not automatically important '
        'for short primary-key transactions; explain applicability using actual workload evidence. '
        'Keep sparse groups tied to a coherent joint performance mechanism. Uncovered active knobs '
        'receive independent coverage candidates; do not pad groups to cover all active knobs. '
        'For regroup replace, propose at least one genuinely different knob combination from current_modes. '
        'The controller selects a pair by joint expected best improvement, accounting for overlap.'
    )

    @staticmethod
    def validate(answer, context):
        PaperCoverageProposer.validate(answer, context)
        if context['phase']!='regroup':return
        reviews=answer.get('regroup_reviews',{})
        owned={key for key,row in context.get('regroup_evidence',{}).items() if row['expert']==context['expert']}
        if not any(reviews[key].get('action')=='replace' for key in owned):return
        previous={tuple(sorted(m['knobs'])) for m in context.get('current_modes',[])}
        if not any(tuple(sorted(g['knobs'])) not in previous for g in answer.get('groups',[])):
            raise ValueError('replace requires a genuinely new combination; otherwise choose continue')
