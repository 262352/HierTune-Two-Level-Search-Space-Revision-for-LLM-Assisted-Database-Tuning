"""Small revisions to coverage-v1; separately versioned and never DB-replayed.

Allocation normalizes current mean NEI and gradually increases coverage weight.
Structural comparisons use regional maxima before positive-part clipping.
Active-set admission includes candidate generation.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from .coverage_search import CoverageSearch,CoverageSettings
from .epoch_search import joint_gains
from cognitune.agents.ownership import EXPERTS


@dataclass(frozen=True)
class RevisedCoverageSettings(CoverageSettings):
    # Replace the old utility-valued additive lambda by dimensionless weights.
    coverage_weight_start:float=.5
    coverage_weight_end:float=2.


def preference_from_draws(old,new,settings,anchor=None):
    """Compare paired untruncated regional maxima using the existing delta/rho.

    A region can be credibly better than another even if both maxima remain
    below the incumbent. The opportunity statistic V is still reported in
    its original positive-part form for macro and alternative ranking.
    """
    old=np.asarray(old,float);new=np.asarray(new,float)
    if old.shape!=new.shape or old.ndim!=1 or not len(old):raise ValueError('Paired regional posterior maxima required')
    anchor=np.zeros_like(old) if anchor is None else np.asarray(anchor,float)
    difference=new-old;mean=float(difference.mean())
    q=float(np.mean(difference>settings.delta));accepted=q>settings.confidence
    old_gain=np.maximum(old-anchor-settings.delta,0.)
    new_gain=np.maximum(new-anchor-settings.delta,0.)
    return {'available':True,'probability':q,'comparison':'untruncated_regional_maximum',
            'clipped_preference_probability':float(np.mean(new_gain>old_gain)),
            'joint_zero_opportunity_fraction':float(np.mean(np.maximum(old_gain,new_gain)==0)),
            'old_value':float(old_gain.mean()),'new_value':float(new_gain.mean()),
            'old_regional_maximum_mean':float(old.mean()),'new_regional_maximum_mean':float(new.mean()),
            'difference_mean':mean,'minimum_regional_advantage':settings.delta,
            'difference_mc_se':float(difference.std(ddof=1)/np.sqrt(len(old))) if len(old)>1 else 0.,
            'probability_mc_se':float(np.sqrt(q*(1-q)/len(old))),
            'accepted':bool(accepted),
            'reason':'accepted' if accepted else 'regional_advantage_not_supported'}


class RevisedCoverageSearch(CoverageSearch):
    allocation_policy='normalized_mean_NEI_times_coverage_with_budget_schedule'

    def __init__(self,*args,**kwargs):
        kwargs.setdefault('settings',RevisedCoverageSettings())
        super().__init__(*args,**kwargs)
        self.state.update(allocation_progress=0.,coverage_repairs=[])

    def set_round_context(self,completed,total):
        self.state['allocation_progress']=float(np.clip(completed/max(total-1,1),0.,1.))

    def allocation_context(self,means):
        progress=self.state['allocation_progress']
        weight=self.settings.coverage_weight_start+(self.settings.coverage_weight_end-self.settings.coverage_weight_start)*progress
        return {'nei_scale':max(means.values(),default=0.),'coverage_weight':weight,
                'adaptive_budget_fraction':progress}

    @staticmethod
    def allocation_weight(mean,context):
        scale=context['nei_scale']
        normalized=mean/scale if scale>0 else 0.
        return float(context['coverage_weight']+normalized)

    def compare_pools(self,gp,anchor,old,new,rng):
        n=min(len(old),len(new))
        if not n:return {'available':False,'accepted':False,'probability':.5,'count':0,'old_value':0.,'new_value':0.}
        _,draws,indices=joint_gains(gp,anchor,[old[:n],new[:n]],rng,self.settings.posterior_samples,self.settings.delta)
        return {**preference_from_draws(draws[:,indices[0]].max(1),draws[:,indices[1]].max(1),
                                     self.settings,draws[:,0]),'count':n}

    def build_modes(self,active,answers,anchor,previous=None,entry=None):
        result=super().build_modes(active,answers,anchor,previous,entry)
        return self.ensure_coverage(result,set(active),answers,anchor)

    def ensure_coverage(self,modes,active,answers,anchor):
        """Ensure all admitted active knobs belong to a generative group.

        Keep two groups/expert and four knobs/group. Fill unused group capacity
        using domain ownership and suggested partners. Repack only if overlap
        has exhausted every slot; this is never a reserved real execution.
        """
        modes=copy.deepcopy(modes);changes=[]
        missing=active-set(n for m in modes.values() for n in m['knobs'])
        for name in sorted(missing,key=lambda n:(-self.state['scores'].get(n,0),n)):
            if any(name in m['knobs'] for m in modes.values()):continue
            block=self.closure([name])
            if len(block)>self.settings.max_group_knobs:raise ValueError('Dependency block exceeds sparse-group capacity: '+name)
            options=[]
            for expert in EXPERTS:
                owned=[(key,m) for key,m in modes.items() if m['expert']==expert]
                for key,m in owned:
                    merged=self.closure(set(m['knobs'])|block)
                    if merged<=active and len(merged)<=self.settings.max_group_knobs:
                        options.append((self.ownership[name].get(expert,0.),1,key,expert,merged))
                if len(owned)<2:
                    key=next(f'{expert}:{i}' for i in range(2) if f'{expert}:{i}' not in modes)
                    partners=answers.get(expert,{}).get('partners',{}).get(name,[])
                    group=set(block)
                    for partner in partners:
                        proposed=self.closure(group|{partner})
                        if proposed<=active and len(proposed)<=self.settings.max_group_knobs:group=proposed
                    options.append((self.ownership[name].get(expert,0.),0,key,expert,group))
            if not options:
                return self.repack_coverage(active,answers,anchor)
            _,_,key,expert,group=max(options,key=lambda x:(x[0],x[1],x[2]))
            old=modes.get(key)
            new=self.make_mode(expert,group,answers.get(expert,{}),anchor,key)
            if old:
                for n in old['knobs']:new['regions'][n]=copy.deepcopy(old['regions'][n])
            modes[key]=new;changes.append({'knob':name,'direction':key,'group':new['knobs']})
        uncovered=active-set(n for m in modes.values() for n in m['knobs'])
        if uncovered:raise RuntimeError('Active knob has no candidate-generating group')
        if changes:
            self.state.setdefault('coverage_repairs',[]).append({'kind':'complete_active_coverage','changes':changes})
            self.state['coverage_repairs']=self.state['coverage_repairs'][-8:]
        return dict(sorted(modes.items()))

    def repack_coverage(self,active,answers,anchor):
        remaining=set(active);result={};slots={e:0 for e in EXPERTS}
        while remaining:
            choices=[]
            for name in remaining:
                for expert in EXPERTS:
                    if slots[expert]<2:choices.append((self.ownership[name].get(expert,0),name,expert))
            if not choices:raise RuntimeError('Active set exceeds candidate-group capacity')
            _,name,expert=max(choices);group=self.closure([name])
            for partner in sorted(remaining-{name},key=lambda n:(-self.ownership[n].get(expert,0),n)):
                proposed=self.closure(group|{partner})
                if proposed<=active and len(proposed)<=self.settings.max_group_knobs:group=proposed
            if not group<=active or len(group)>self.settings.max_group_knobs:
                raise RuntimeError('Cannot cover active dependency block')
            key=f'{expert}:{slots[expert]}';slots[expert]+=1
            result[key]=self.make_mode(expert,group,answers.get(expert,{}),anchor,key)
            remaining-=group
        self.state.setdefault('coverage_repairs',[]).append({'kind':'repack_overlapping_groups','groups':{k:m['knobs'] for k,m in result.items()}})
        self.state['coverage_repairs']=self.state['coverage_repairs'][-8:]
        return dict(sorted(result.items()))

    def micro(self,gp,anchor,rng,seen):
        reports=super().micro(gp,anchor,rng,seen)
        self.state['modes']=self.ensure_coverage(self.state['modes'],set(self.state['active']),self.state['proposals'],anchor)
        return reports

    def order_for_exposure(self,mode,rows,anchor,rng,seen):
        """Place legal variations in the initial quota; do not force execution."""
        rows=list(rows);front=[];remaining=set(mode['knobs'])
        used={self.space.config_key(c) for c in rows}|set(seen)
        for name in mode['knobs']:
            if not any(c.get(name)!=anchor.get(name) for c in rows):
                rows+=self.sample(mode,anchor,rng,self.settings.probes,used,full=True)
                used.update(self.space.config_key(c) for c in rows)
        candidates=list(rows)
        while remaining and candidates and len(front)<self.settings.initial_candidates:
            best=max(candidates,key=lambda c:sum(c.get(n)!=anchor.get(n) for n in remaining))
            varied={n for n in remaining if best.get(n)!=anchor.get(n)}
            if not varied:break
            front.append(best);candidates.remove(best);remaining-=varied
        return front+candidates

    def candidate_pool(self,*args,**kwargs):
        configs,sources,report=super().candidate_pool(*args,**kwargs)
        report.pop('lambda',None)
        report['coverage_repairs']=copy.deepcopy(self.state['coverage_repairs'])
        report['unexposed_active_knobs']=[n for n,c in report['active_knob_candidate_exposure'].items() if not c]
        return configs,sources,report
