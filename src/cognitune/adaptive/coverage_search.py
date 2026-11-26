"""Paper-method variant: local revisions, tolerant macro removal and NEI/coverage.

All allocation probes are model-only. Diagnostics distinguish geometric upper
bounds from empirical feasible diameters, and generated reservoirs from the
configurations actually admitted to the fixed-capacity candidate pool.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist
from .epoch_search import EpochSearch, EpochSettings, joint_gains, compare
from .nested_search import strict_patch, inside_region
from cognitune.agents.ownership import EXPERTS


@dataclass(frozen=True)
class CoverageSettings(EpochSettings):
    max_incoming:int=5
    initial_candidates:int=4
    allocation_floor:float=.02
    global_probability:float=.1
    stale_executions:int=6
    low_opportunity_executions:int=3
    low_opportunity_probability:float=.1
    max_group_knobs:int=4
    entrants_per_expert:int=6


class CoverageSearch(EpochSearch):
    def __init__(self,*args,ownership=None,**kwargs):
        kwargs.setdefault('settings',CoverageSettings())
        super().__init__(*args,**kwargs)
        if ownership is not None:self.ownership=copy.deepcopy(ownership)
        self.state.update(local={},regroup_requests={},reference_regions={},ever_changed=[],manager_weights={})

    def expert_context(self,expert,phase):
        return {'local_execution_evidence':self.state['local'],
                'regroup_evidence':self.state['regroup_requests'],
                'reference_regions':self.state['reference_regions'],
                'score_inactive_knobs':phase=='macro',
                'max_group_knobs':self.settings.max_group_knobs}

    def initialize(self,answers,anchor):
        # The runner supplies manager-derived ownership before initialization.
        result=super().initialize(answers,anchor)
        self.state['manager_weights']=copy.deepcopy(self.ownership)
        self.update_reference(answers)
        return result

    def update_reference(self,answers):
        for name in self.state['active']:
            opinions=[(w,e) for e,w in self.ownership[name].items()
                      if name in answers.get(e,{}).get('regions',{})]
            if opinions:
                _,expert=max(opinions)
                self.state['reference_regions'][name]=copy.deepcopy(answers[expert]['regions'][name])

    def build_modes(self,active,answers,anchor,previous=None,entry=None):
        # Do not silently project a sparse group to a different interaction.
        filtered=copy.deepcopy(answers)
        for a in filtered.values():
            a['groups']=[g for g in a.get('groups',[]) if
                         0<len(self.closure(g.get('knobs',[])))<=self.settings.max_group_knobs]
            a['patches']=[p for p in a.get('patches',[]) if
                          0<len(self.closure(p))<=self.settings.max_group_knobs]
        result=super().build_modes(active,filtered,anchor,previous,entry)
        return {k:m for k,m in result.items() if len(m['knobs'])<=self.settings.max_group_knobs}

    def make_mode(self,expert,names,answer,anchor,key):
        # Overall reference ranges supply defaults; local recommendations win.
        a=copy.deepcopy(answer)
        a['regions']={**copy.deepcopy(self.state.get('reference_regions',{})),**a.get('regions',{})}
        return super().make_mode(expert,names,a,anchor,key)

    def sample(self,mode,anchor,rng,n,seen=(),outside=False,full=False):
        if n<=0:return []
        seen=set(seen);out=[];local=set();group=mode['knobs']
        for _ in range(max(256,n*100)):
            if len(out)>=n:break
            global_draw=full or outside or (getattr(self,'_candidate_sampling',False) and rng.random()<self.settings.global_probability)
            patch={}
            for name in group:
                k=self.space.by_name[name];r=mode['regions'][name]
                if k.kind in {'integer','float'}:
                    if global_draw:z=rng.uniform(0.,1.)
                    else:
                        lo=max(0.,r['center']-r['halfwidth']);hi=min(1.,r['center']+r['halfwidth'])
                        z=None
                        for _ in range(80):
                            value=rng.normal(r['center'],max(r['halfwidth']/2,1e-8))
                            if lo<=value<=hi:z=value;break
                        if z is None:z=rng.uniform(lo,hi)
                    # Half of external draws probe a narrow band just outside
                    # the old boundary; the other half use the full legal box.
                    if outside and rng.random()<.5:
                        edge=r['center']+rng.choice([-1,1])*r['halfwidth']
                        z=float(np.clip(edge+rng.uniform(-.25,.25)*r['halfwidth'],0,1))
                    patch[name]=self.value_at(k,z)
                else:
                    values=([False,True] if k.kind=='bool' else list(k.choices)) if global_draw else r['values']
                    patch[name]=values[int(rng.integers(len(values)))]
            cfg=strict_patch(self.space,anchor,patch,group)
            if cfg is None:continue
            if outside and inside_region(self.space,cfg,group,mode['regions'],self.unit):continue
            key=self.space.config_key(cfg)
            if key not in seen and key not in local:
                out.append(cfg);local.add(key)
        return out

    def local_record(self,key,mode):
        signature=repr((mode['knobs'],mode['regions']))
        record=self.state['local'].get(key)
        if record is None or record['signature']!=signature:
            record={'signature':signature,'executions':0,'stale':0,'last_review_stale':-1}
            self.state['local'][key]=record
        return record

    def observe_result(self,plan,result):
        self.observe(plan['anchor'],plan['config'],result.status,plan['trial'])
        self.state['ever_changed']=sorted(set(self.state['ever_changed'])|
                    set(self.state['executions'][-1]['changed_knobs']))
        for key in set(plan.get('sources',[])):
            mode=self.state['modes'].get(key)
            if not mode:continue
            record=self.local_record(key,mode);record['executions']+=1
            improvement=result.status=='SUCCESS' and result.utility-plan['anchor_utility']>self.settings.delta
            record['stale']=0 if improvement else record['stale']+1
            record['last_trial']=plan['trial'];record['last_improvement']=bool(improvement)

    def regroup_requests(self,gp,anchor,rng,seen):
        due={}
        for key,mode in sorted(self.state['modes'].items()):
            record=self.local_record(key,mode)
            if record['stale']<self.settings.low_opportunity_executions:continue
            # One request per new local execution, not every unselected round.
            if record['last_review_stale']==record['stale']:continue
            probes=self.sample(mode,anchor,rng,self.settings.probes,seen)
            gains,_,_=joint_gains(gp,anchor,[probes],rng,self.settings.posterior_samples,self.settings.delta)
            p=float(np.mean(gains[0]>0))
            trigger=record['stale']>=self.settings.stale_executions or p<=self.settings.low_opportunity_probability
            if trigger:
                record['last_review_stale']=record['stale']
                due[key]={'expert':mode['expert'],'local_executions':record['executions'],
                          'stale_local_executions':record['stale'],'opportunity_probability':p,
                          'reason':'six_stale' if record['stale']>=self.settings.stale_executions else 'three_stale_low_probability'}
        self.state['regroup_requests']=due
        return {r['expert'] for r in due.values()}

    def micro(self,gp,anchor,rng,seen):
        result={};reports={}
        for key,mode in sorted(self.state['modes'].items()):
            result[key],reports[key]=self.micro_mode(gp,anchor,mode,rng,seen)
            self.local_record(key,result[key])
            if key not in self.state['regroup_requests']:continue
            expert=mode['expert'];proposals=self.state['proposals'].get(expert,{})
            accepted=[];comparisons=[]
            for g in proposals.get('groups',[])[:2]:
                names=self.closure(g['knobs'])
                if not names<=set(self.state['active']) or len(names)>self.settings.max_group_knobs:continue
                if names==set(mode['knobs']):continue
                candidate=self.make_mode(expert,names,proposals,anchor,key)
                old=self.sample(result[key],anchor,rng,self.settings.probes,seen)
                new=self.sample(candidate,anchor,rng,self.settings.probes,seen)
                cmp=(self.compare_pools(gp,anchor,old,new,rng) if hasattr(self,'compare_pools')
                     else compare(gp,anchor,old,new,rng,self.settings))
                comparisons.append({'knobs':sorted(names),**cmp})
                if cmp.get('accepted',cmp['probability']>self.settings.confidence):accepted.append((cmp['new_value'],candidate))
            reports[key]['regroup']={'trigger':self.state['regroup_requests'][key],'comparisons':comparisons}
            if accepted:
                _,result[key]=max(accepted,key=lambda x:x[0]);self.local_record(key,result[key])
                reports[key]['group_change']={'from':mode['knobs'],'to':result[key]['knobs']}
        self.state['modes']=result
        return reports

    def macro(self,gp,anchor,rng,seen,adaptive_completed,answers):
        self.state['last_macro']=adaptive_completed
        old_active=set(self.state['active']);active=set(old_active)
        modes=copy.deepcopy(self.state['modes']);removed=[];checks=[]
        eligible=set(self.state['ever_changed'])&active
        for _ in range(self.settings.max_incoming):
            base=self.state_pool(modes,anchor,rng,seen);options=[]
            for name in sorted(eligible&active):
                block={name};changed=True
                while changed:
                    expanded={k.name for k in self.space.knobs if k.condition and k.condition.parent in block and k.name in active}
                    changed=not expanded<=block;block|=expanded
                if len(removed)+len(block)>self.settings.max_incoming or not active-block:continue
                reduced=self.build_modes(active-block,answers,anchor,modes)
                proposed=self.state_pool(reduced,anchor,rng,seen);n=min(len(base),len(proposed))
                if not n:continue
                gains,_,_=joint_gains(gp,anchor,[base[:n],proposed[:n]],rng,self.settings.posterior_samples,self.settings.delta)
                loss=float(np.mean(gains[0]-gains[1]));support=float(np.mean(gains[1]>=gains[0]-self.settings.delta))
                accepted=loss<=self.settings.delta and support>self.settings.confidence
                row={'knobs':sorted(block),'loss':loss,'support':support,'count':n,'accepted':accepted}
                checks.append(row)
                if accepted:options.append((loss,sorted(block),reduced))
            if not options:break
            _,block,modes=min(options,key=lambda a:(a[0],a[1]));active-=set(block);removed+=block
        # Latest inactive scores are aggregated with the fixed manager weights.
        old_scores=self.state['scores'];latest=dict(old_scores);nominations=set()
        for expert in EXPERTS:
            eligible_names=[n for n in self.space.by_name if n not in old_active and expert in self.ownership[n]]
            scores=answers.get(expert,{}).get('scores',{})
            ranked=sorted(eligible_names,key=lambda n:(-scores.get(n,old_scores.get(n,0.)),n))[:self.settings.entrants_per_expert]
            nominations.update(ranked)
        for n in nominations:
            latest[n]=sum(w*answers.get(e,{}).get('scores',{}).get(n,old_scores.get(n,0.)) for e,w in self.ownership[n].items())
        entrants=[]
        for name in sorted(nominations,key=lambda n:(-latest[n],n)):
            closure=self.closure([name]);added=closure-active
            if len(active|closure)>self.active_target or len(entrants)+len(added)>len(removed):continue
            active|=closure;entrants+=sorted(added)
            if len(entrants)>=len(removed):break
        self.state.update(active=sorted(active),scores=latest,proposals=copy.deepcopy(answers))
        # Refresh overall ranges at macro review, including retained knobs.
        before=copy.deepcopy(self.state['reference_regions']);self.update_reference(answers)
        changed_ranges={n for n in active if before.get(n)!=self.state['reference_regions'].get(n)}
        affected=set(removed)|set(entrants)|changed_ranges
        preserved={k:m for k,m in modes.items() if not set(m['knobs'])&affected}
        rebuilt=self.build_modes(active,answers,anchor,preserved)
        # Preserve unrelated exact combinations; affected combinations inherit
        # the new expert regions rather than local regions from the old state.
        for key,m in rebuilt.items():
            old=next((m0 for m0 in preserved.values() if m0['expert']==m['expert'] and m0['knobs']==m['knobs']),None)
            if old is not None:rebuilt[key]=copy.deepcopy(old);rebuilt[key]['id']=key
        self.state['modes']=rebuilt
        return {'review':True,'adaptive_completed':adaptive_completed,'accepted':active!=old_active or bool(changed_ranges),
                'eligible':sorted(eligible),'removal':checks,'removed':removed,'entered':entrants,
                'entry_ranking':sorted([{'knob':n,'score':latest[n]} for n in nominations],key=lambda a:(-a['score'],a['knob'])),
                'reference_changes':sorted(changed_ranges),'active_after':sorted(active)}

    def diameter(self,gp,mode,anchor,pool):
        lengths=np.asarray(gp.model.covar_module.base_kernel.lengthscale.detach().cpu()).reshape(-1)
        bounds=[];coordinate_indices=[]
        for node in self.space.encoder.nodes:
            if node.knob.name not in mode['knobs']:continue
            k=node.knob;r=mode['regions'][k.name]
            indices=list(range(node.value_start,node.value_stop));coordinate_indices+=indices
            if k.kind in {'integer','float'}:
                span=min(1.,r['center']+r['halfwidth'])-max(0.,r['center']-r['halfwidth'])
                bounds.append((span/lengths[indices[0]])**2)
            elif k.kind=='ordinal':
                z=[k.choices.index(v)/(len(k.choices)-1) for v in r['values']]
                bounds.append(((max(z)-min(z))/lengths[indices[0]])**2)
            else:
                choices=[False,True] if k.kind=='bool' else list(k.choices)
                ids=[indices[choices.index(v)] for v in r['values']]
                inverse=sorted([1/lengths[i]**2 for i in ids],reverse=True)
                bounds.append(sum(inverse[:2]) if len(inverse)>1 else 0.)
        box=float(np.sqrt(sum(bounds)))
        encoded=self.space.encode_many(pool)[:,coordinate_indices]/lengths[coordinate_indices] if pool else np.empty((0,0))
        empirical=float(pdist(encoded).max()) if len(pool)>1 else 0.
        constrained=bool(self.space.forbidden or self.space.hard_constraints or any(self.space.by_name[n].condition for n in mode['knobs']))
        # In constrained mixed spaces an exact maximum is expensive. Report
        # the feasible finite-set approximation and the box upper bound.
        return (empirical if constrained else box),{'box_upper_bound':box,'feasible_empirical_diameter':empirical,
                    'approximation':'feasible_pool_max' if constrained else 'independent_region_box',
                    'knob_dimension':len(mode['knobs']),'encoded_value_dimensions':len(coordinate_indices)}

    def candidate_pool(self,gp,anchor,rng,seen,total=100,audit_size=4):
        modes=self.state['modes'];initial=self.settings.initial_candidates
        reservoirs={};origins={};excluded=set(seen)
        # Draw a bounded reservoir per direction. Allocation only admits the
        # first n_h points; unadmitted probes are never counted in the pool.
        self._candidate_sampling=True
        for key,mode in sorted(modes.items()):
            rows=[];used=set(excluded)
            for patch in self.state['proposals'].get(mode['expert'],{}).get('patches',[]):
                if not patch or not set(patch)<=set(mode['knobs']):continue
                cfg=strict_patch(self.space,anchor,patch,mode['knobs'])
                if cfg is not None and self.space.config_key(cfg) not in used:
                    rows.append(cfg);used.add(self.space.config_key(cfg));origins[self.space.config_key(cfg)]='expert_patch'
            rows+=self.sample(mode,anchor,rng,max(0,total-len(rows)),used)
            if hasattr(self,'order_for_exposure'):
                rows=self.order_for_exposure(mode,rows,anchor,rng,used)
            reservoirs[key]=rows[:total]
        self._candidate_sampling=False
        flat=[c for rows in reservoirs.values() for c in rows]
        scores=np.exp(np.asarray(gp.log_noisy_expected_improvement(flat,samples=self.settings.posterior_samples,seed=int(rng.integers(2**31-1))))) if flat else np.empty(0)
        if not np.isfinite(scores).all():raise ValueError('Non-finite allocation NEI')
        estimates={};scales={};geometries={};counts={};offset=0;pool={};sources={};steps=[]
        for key,rows in reservoirs.items():
            estimates[key]=scores[offset:offset+len(rows)];offset+=len(rows)
            scales[key],geometries[key]=self.diameter(gp,modes[key],anchor,rows)
            counts[key]=0
        def append(key):
            n=counts[key];cfg=reservoirs[key][n];digest=self.space.config_key(cfg)
            counts[key]+=1;pool[digest]=cfg
            if key not in sources.setdefault(digest,[]):sources[digest].append(key)
        for key,rows in reservoirs.items():
            for _ in range(min(initial,len(rows))):append(key)
        if len(pool)>total:raise ValueError('Initial subspace candidates exceed pool capacity')
        for _ in range(total*max(1,len(modes))):
            if len(pool)>=total:break
            priorities={}
            means={key:float(estimates[key][:counts[key]].mean()) for key in modes if counts[key]}
            context=self.allocation_context(means) if hasattr(self,'allocation_context') else {}
            for key,rows in reservoirs.items():
                n=counts[key]
                if not n or n>=len(rows):continue
                d=len(modes[key]['knobs']);mean=float(estimates[key][:n].mean())
                marginal=scales[key]*(n**(-1/d)-(n+1)**(-1/d))
                weight=(self.allocation_weight(mean,context) if hasattr(self,'allocation_weight')
                        else self.settings.allocation_floor+mean)
                priorities[key]=weight*marginal
            if not priorities:break
            key=max(priorities,key=lambda k:(priorities[k],k));steps.append({'direction':key,'n_before':counts[key],
                'mean_nei':float(estimates[key][:counts[key]].mean()),'priority':priorities[key],**context})
            append(key)
        if not pool:raise RuntimeError('No legal untested candidates')
        exposure={n:sum(c.get(n)!=anchor.get(n) for c in pool.values()) for n in self.state['active']}
        return list(pool.values()),sources,{'policy':getattr(self,'allocation_policy','progressive_mean_NEI_times_coverage_marginal'),'capacity':total,
            'pool_size':len(pool),'quotas':counts,'geometry':geometries,'scale':scales,
            'mean_nei':{k:float(estimates[k][:n].mean()) for k,n in counts.items() if n},
            'reservoir_counts':{k:len(v) for k,v in reservoirs.items()},'allocation_steps':steps,
            'initial_candidates':initial,'lambda':self.settings.allocation_floor,
            'patch_candidates_admitted':sum(k in origins for k in pool),
            'candidate_sources':sources,'active_knob_candidate_exposure':exposure,
            'unavailable_directions':[k for k,v in reservoirs.items() if not v]}
