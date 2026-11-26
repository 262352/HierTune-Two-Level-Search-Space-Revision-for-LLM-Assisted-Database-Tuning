"""Posterior-location micro revision and twelve-round active-set revision.

Audits are finite model calculations, never database observations. This policy
has no residence, forced re-entry, calibration or reserved real-trial gates.
"""
from __future__ import annotations
import copy
from dataclasses import asdict,dataclass
import numpy as np
from .decision import factor
from .nested_search import strict_patch,inside_region
from cognitune.agents.ownership import EXPERTS,build_ownership

@dataclass(frozen=True)
class EpochSettings:
    macro_interval:int=12
    probes:int=16
    posterior_samples:int=256
    confidence:float=.8
    location_mass:float=.8
    delta:float=.01
    max_incoming:int=2
    donor_shortlist:int=2
    incoming_shortlist:int=3

def joint_gains(gp,anchor,pools,rng,samples=256,delta=.01):
    """One correlated function bank for overlapping, frozen finite pools."""
    configs=[anchor]; indices={repr(sorted(anchor.items())):0}; slices=[]
    for pool in pools:
        row=[]
        for cfg in pool:
            key=repr(sorted(cfg.items()))
            if key not in indices:
                indices[key]=len(configs);configs.append(cfg)
            row.append(indices[key])
        slices.append(row)
    mean,cov,_=gp.joint_posterior(configs)
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError('Non-finite structural posterior')
    draws=np.asarray(mean)+rng.standard_normal((samples,len(configs)))@factor(np.asarray(cov)).T
    gains=[np.maximum(draws[:,ix].max(1)-draws[:,0]-delta,0.) if ix else np.zeros(samples)
           for ix in slices]
    return gains,draws,slices

def compare(gp,anchor,old,new,rng,settings):
    n=min(len(old),len(new))
    if not n:return {'available':False,'probability':0.,'count':0,'old_value':0.,'new_value':0.}
    gains,_,_=joint_gains(gp,anchor,[old[:n],new[:n]],rng,settings.posterior_samples,settings.delta)
    p=float(np.mean(gains[1]>gains[0]))
    return {'available':True,'count':n,'probability':p,'old_value':float(gains[0].mean()),
            'new_value':float(gains[1].mean()),'difference_mean':float((gains[1]-gains[0]).mean()),
            'probability_mc_se':float(np.sqrt(p*(1-p)/settings.posterior_samples))}

def weighted_quantile(values,weights,mass):
    order=np.argsort(values,kind='stable')
    pos=np.searchsorted(np.cumsum(np.asarray(weights)[order]),mass*np.sum(weights))
    return float(np.asarray(values)[order[min(pos,len(order)-1)]])

def geometry(space,mode,configs,weights,unit,settings):
    """A legal joint medoid and ONE posterior-computed scale multiplier."""
    weights=np.asarray(weights,float)
    if not len(configs) or weights.sum()<=0:return None
    weights=weights/weights.sum()
    names=mode['knobs'];old=mode['regions']
    distances=np.zeros((len(configs),len(configs)))
    categorical=np.zeros_like(distances)
    numeric=[n for n in names if space.by_name[n].kind in {'integer','float'}]
    for name in names:
        k=space.by_name[name]
        if name in numeric:
            z=np.array([unit(k,c[name]) for c in configs])
            distances=np.maximum(distances,np.abs(z[:,None]-z[None,:])/max(old[name]['halfwidth'],1e-9))
        else:
            z=[c[name] for c in configs]
            categorical+=np.array([[a!=b for b in z] for a in z],float)
    distances+=categorical
    chosen=int(np.argmin(distances@weights))
    result=copy.deepcopy(mode);center=configs[chosen]
    scale=1.
    if numeric:
        d=np.max(np.array([np.abs(np.array([unit(space.by_name[n],c[n]) for c in configs])
                       -unit(space.by_name[n],center[n]))/max(old[n]['halfwidth'],1e-9)
                       for n in numeric]),axis=0)
        scale=float(np.clip(weighted_quantile(d,weights,settings.location_mass),.5,2.))
    for name in names:
        k=space.by_name[name]
        if name in numeric:
            result['regions'][name]={'center':float(unit(k,center[name])),
                 'halfwidth':float(min(1.,max(1e-8,old[name]['halfwidth']*scale)))}
        else:
            choices=list(dict.fromkeys(c[name] for c in configs))
            masses=[sum(w for c,w in zip(configs,weights) if c[name]==v) for v in choices]
            order=sorted(range(len(choices)),key=lambda i:-masses[i])
            keep=[];total=0.
            for i in order:
                keep.append(choices[i]);total+=masses[i]
                if total>=settings.location_mass:break
            result['regions'][name]={'values':keep}
    return result,{'center_config':copy.deepcopy(center),'scale':scale,
                   'location_mass':settings.location_mass}

class EpochSearch:
    def __init__(self,space,unit,value_at,*,settings=None,active_target=18):
        self.space,self.unit,self.value_at=space,unit,value_at
        self.settings=settings or EpochSettings()
        self.ownership=build_ownership(space);self.active_target=active_target
        self.state={'active':[],'modes':{},'proposals':{},'scores':{},'last_macro':0,'executions':[]}

    def closure(self,group):
        names=set(group)
        for name in tuple(names):
            k=self.space.by_name[name]
            while k.condition:
                names.add(k.condition.parent);k=self.space.by_name[k.condition.parent]
        return names

    def initialize(self,answers,anchor):
        scores={n:sum(weights[e]*answers[e]['scores'][n] for e in weights)
                for n,weights in self.ownership.items()}
        active=set()
        # Initial domain coverage; later revisions are opportunity based.
        for expert in EXPERTS:
            ranked=sorted((n for n in scores if expert in self.ownership[n]),key=lambda n:(-scores[n],n))
            for name in ranked:
                closure=self.closure([name])
                if len(active|closure)<=self.active_target:
                    active.update(closure);break
        for name in sorted(scores,key=lambda n:(-scores[n],n)):
            if len(active|self.closure([name]))<=self.active_target:active.update(self.closure([name]))
        self.state.update(active=sorted(active),scores=scores,proposals=copy.deepcopy(answers))
        self.state['modes']=self.build_modes(active,answers,anchor)
        return copy.deepcopy(self.state)

    def make_mode(self,expert,names,answer,anchor,key):
        names=sorted(self.closure(names))
        regions={}
        for name in names:
            k=self.space.by_name[name];opinion=answer.get('regions',{}).get(name,{})
            if k.kind in {'integer','float'}:
                regions[name]={'center':float(self.unit(k,opinion.get('center',anchor.get(name,k.default)))),
                               'halfwidth':float(opinion.get('halfwidth',.15))}
            else:
                values=opinion.get('values') or [opinion.get('center',anchor.get(name,k.default))]
                regions[name]={'values':list(dict.fromkeys(values))}
        for name in names:
            k=self.space.by_name[name]
            if k.condition and k.condition.parent in regions:
                parent=self.space.by_name[k.condition.parent]
                if parent.kind not in {'integer','float'}:
                    regions[k.condition.parent]={'values':list(k.condition.values)}
        return {'expert':expert,'knobs':names,'regions':regions,'id':key}

    def build_modes(self,active,answers,anchor,previous=None,entry=None):
        active=set(active);result={}
        for expert in EXPERTS:
            answer=answers.get(expert,{})
            groups=[g['knobs'] for g in answer.get('groups',[]) if set(g.get('knobs',[]))<=active]
            groups += [list(p) for p in answer.get('patches',[]) if p and set(p)<=active]
            old=[m for m in (previous or {}).values() if m['expert']==expert]
            if not groups:
                groups=[sorted(set(m['knobs'])&active) for m in old if set(m['knobs'])&active]
            if not groups:
                owned=sorted((n for n in active if expert in self.ownership[n]),
                             key=lambda n:(-self.state['scores'].get(n,0.),n))
                groups=[owned[:3] or sorted(active)[:1]]
            unique=[]
            for names in groups:
                g=tuple(sorted(self.closure(names)))
                if g and set(g)<=active and g not in unique:unique.append(g)
            if entry and expert==entry[0]:
                g=tuple(sorted(self.closure(entry[1])))
                if set(g)<=active:
                    unique=[g]+[x for x in unique if x!=g]
            for i,names in enumerate(unique[:2]):
                key=f'{expert}:{i}'
                old_mode=next((m for m in old if m['knobs']==list(names)),None)
                result[key]=copy.deepcopy(old_mode) if old_mode else self.make_mode(expert,names,answer,anchor,key)
                result[key]['id']=key
        return {k:result[k] for k in sorted(result)}

    def sample(self,mode,anchor,rng,n,seen=(),outside=False,full=False):
        rows=[];local=set();seen=set(seen);group=mode['knobs']
        if not group:return rows
        for _ in range(max(256,n*80)):
            if len(rows)>=n:break
            patch={}
            for name in group:
                k=self.space.by_name[name];r=mode['regions'][name]
                if k.kind in {'integer','float'}:
                    lo,hi=(0.,1.) if outside or full else (max(0.,r['center']-r['halfwidth']),min(1.,r['center']+r['halfwidth']))
                    patch[name]=self.value_at(k,rng.uniform(lo,hi))
                else:
                    values=([False,True] if k.kind=='bool' else list(k.choices)) if outside or full else r['values']
                    patch[name]=values[int(rng.integers(len(values)))]
            cfg=strict_patch(self.space,anchor,patch,group)
            if cfg is None:continue
            within=inside_region(self.space,cfg,group,mode['regions'],self.unit)
            if not full and (within if outside else not within):continue
            key=self.space.config_key(cfg)
            if key not in seen and key not in local:
                local.add(key);rows.append(cfg)
        return rows

    def value(self,gp,anchor,pool,rng):
        if not pool:return 0.
        gains,_,_=joint_gains(gp,anchor,[pool],rng,self.settings.posterior_samples,self.settings.delta)
        return float(gains[0].mean())

    def micro_mode(self,gp,anchor,mode,rng,seen):
        n=self.settings.probes
        inside=self.sample(mode,anchor,rng,n,seen)
        outside=self.sample(mode,anchor,rng,n,seen,outside=True)
        count=min(len(inside),len(outside))
        # A full recommendation has no outside; it can still refine internally.
        pools=[inside[:count],outside[:count]] if count else [inside,outside]
        both=list({self.space.config_key(c):c for rows in pools for c in rows}.values())
        if not both:return copy.deepcopy(mode),{'accepted':False,'reason':'no_legal_probes'}
        gains,draws,ix=joint_gains(gp,anchor,[both],rng,self.settings.posterior_samples,self.settings.delta)
        winner=np.argmax(draws[:,ix[0]],axis=1)
        weights=np.bincount(winner,weights=gains[0],minlength=len(both))
        proposal=geometry(self.space,mode,both,weights,self.unit,self.settings)
        report={'inside_count':len(pools[0]),'outside_count':len(pools[1]),
                'inside_value':self.value(gp,anchor,pools[0],rng),
                'outside_value':self.value(gp,anchor,pools[1],rng)}
        if proposal is None:
            return copy.deepcopy(mode),{**report,'accepted':False,'reason':'no_improving_draws'}
        new,geo=proposal
        old_pool=self.sample(mode,anchor,rng,n,seen)
        new_pool=self.sample(new,anchor,rng,n,seen)
        audit=(self.compare_pools(gp,anchor,old_pool,new_pool,rng) if hasattr(self,'compare_pools')
               else compare(gp,anchor,old_pool,new_pool,rng,self.settings))
        accepted=audit.get('accepted',audit['available'] and audit['probability']>self.settings.confidence)
        return new if accepted else copy.deepcopy(mode),{**report,**geo,**audit,'accepted':accepted,
             'before':mode['regions'],'proposed':new['regions']}

    def micro(self,gp,anchor,rng,seen):
        result={};reports={}
        for key,mode in sorted(self.state['modes'].items()):
            result[key],reports[key]=self.micro_mode(gp,anchor,mode,rng,seen)
        # Cached alternative expert groups are genuine proposals, not invented API feedback.
        alternatives=self.build_modes(self.state['active'],self.state['proposals'],anchor)
        for key,mode in sorted(alternatives.items()):
            old=result.get(key)
            if old is None or old['knobs']==mode['knobs']:continue
            old_pool=self.sample(old,anchor,rng,self.settings.probes,seen)
            new_pool=self.sample(mode,anchor,rng,self.settings.probes,seen)
            cmp=compare(gp,anchor,old_pool,new_pool,rng,self.settings)
            if cmp['probability']>self.settings.confidence:
                result[key]=mode;reports[key]['group_change']={'from':old['knobs'],'to':mode['knobs'],**cmp}
        self.state['modes']=result
        return reports

    def state_pool(self,modes,anchor,rng,seen):
        """Four fixed 16-point mixtures. No current quotas or duplicated filler."""
        per_expert={};local=set(seen)
        for expert in EXPERTS:
            choices=[m for _,m in sorted(modes.items()) if m['expert']==expert]
            if not choices:return []
            own=[]
            for turn in range(64):
                if len(own)>=16:break
                mode=choices[turn%len(choices)]
                batch=self.sample(mode,anchor,rng,1,local)
                if batch:
                    key=self.space.config_key(batch[0]);local.add(key);own+=batch
            per_expert[expert]=own
        m=min(map(len,per_expert.values()))
        # Round-robin ordering preserves equal expert shares when two states
        # must use their common attainable capacity instead of 64 points.
        return [per_expert[e][i] for i in range(m) for e in EXPERTS]

    def review_due(self,adaptive_completed):
        return adaptive_completed>0 and adaptive_completed-self.state['last_macro']>=self.settings.macro_interval

    def paired_entry(self,new,old,anchor,rng,seen):
        """Same partner coordinates; insertion differs only in the new block."""
        before=[];after=[];old_keys=set();new_keys=set()
        for cfg in self.sample(new,anchor,rng,self.settings.probes*8,seen,full=True):
            patch={n:cfg[n] for n in old['knobs'] if n in cfg}
            frozen=strict_patch(self.space,anchor,patch,old['knobs'])
            if frozen is None:continue
            k0,k1=self.space.config_key(frozen),self.space.config_key(cfg)
            if k0 in seen or k0 in old_keys or k1 in new_keys:continue
            old_keys.add(k0);new_keys.add(k1);before.append(frozen);after.append(cfg)
            if len(before)>=self.settings.probes:break
        return before,after

    def macro(self,gp,anchor,rng,seen,adaptive_completed,answers):
        active=set(self.state['active'])
        window=self.state['executions'][-self.settings.macro_interval:]
        eligible=sorted(active&{n for row in window if row['status']=='SUCCESS' for n in row['changed_knobs']})
        report={'review':True,'adaptive_completed':adaptive_completed,'eligible':eligible,
                'removal':[],'entry':[],'alternatives':[],'accepted':False,'swaps':[]}
        self.state['last_macro']=adaptive_completed
        self.state['proposals']=copy.deepcopy(answers)
        original=copy.deepcopy(self.state['modes'])
        base=self.state_pool(original,anchor,rng,seen)
        if not base or not eligible:return report
        donors=[]
        for name in eligible:
            block={name}
            changed=True
            while changed:
                grown={k.name for k in self.space.knobs if k.condition and k.condition.parent in block and k.name in active}
                changed=not grown<=block;block|=grown
            if len(block)>self.settings.max_incoming or active<=block:continue
            reduced=self.build_modes(active-block,answers,anchor,original)
            pool=self.state_pool(reduced,anchor,rng,seen)
            cmp=compare(gp,anchor,base,pool,rng,self.settings)
            loss=cmp['old_value']-cmp['new_value']
            donors.append((loss,tuple(sorted(block))))
            report['removal'].append({'knobs':sorted(block),'loss':loss,'count':cmp['count']})
        incoming=[]
        for name in sorted(set(self.space.by_name)-active):
            owners=self.ownership[name]
            expert=max(owners,key=owners.get)
            partners=answers[expert].get('partners',{}).get(name,[])
            partners=[n for n in partners if n in active]
            if not partners:
                owned=[n for n in sorted(active) if expert in self.ownership[n]]
                partners=(owned or sorted(active))[:2]
            group=self.closure([name]+partners)
            added=group-active
            if len(added)>self.settings.max_incoming:continue
            new=self.make_mode(expert,group,answers[expert],anchor,'entry')
            old=self.make_mode(expert,self.closure(partners),answers[expert],anchor,'partner')
            old_pool,new_pool=self.paired_entry(new,old,anchor,rng,seen)
            cmp=compare(gp,anchor,old_pool,new_pool,rng,self.settings)
            if not cmp['available']:continue
            gain=cmp['new_value']-cmp['old_value']
            incoming.append((gain,name,expert,group))
            report['entry'].append({'knob':name,'partners':partners,'gain':gain,'count':cmp['count']})
        donors=sorted(set(donors))[:self.settings.donor_shortlist]
        incoming=sorted(incoming,key=lambda x:(-x[0],x[1]))[:self.settings.incoming_shortlist]
        accepted=[]
        for _,block in donors:
            for _,name,expert,group in incoming:
                if set(block)&group:continue
                candidate=(active-set(block))|group
                if len(candidate)>self.active_target:continue
                modes=self.build_modes(candidate,answers,anchor,original,entry=(expert,group))
                # Recheck all retained domains under the candidate active context.
                for key,m in sorted(modes.items()):
                    modes[key],_=self.micro_mode(gp,anchor,m,rng,seen)
                # Final comparison uses a NEW equal-capacity frozen audit.
                old_pool=self.state_pool(original,anchor,rng,seen)
                new_pool=self.state_pool(modes,anchor,rng,seen)
                cmp=compare(gp,anchor,old_pool,new_pool,rng,self.settings)
                item={'out':list(block),'in':sorted(candidate-active),**cmp}
                report['alternatives'].append(item)
                if cmp['probability']>self.settings.confidence:
                    accepted.append((cmp['new_value'],candidate,modes,item))
        if accepted:
            _,candidate,modes,item=max(accepted,key=lambda x:x[0])
            self.state['active']=sorted(candidate);self.state['modes']=modes
            report.update(accepted=True,swaps=[item])
        return report

    def observe(self,anchor,config,status,trial):
        self.state['executions'].append({'trial':trial,'status':status,
            'anchor':copy.deepcopy(anchor),'changed_knobs':sorted(n for n in config if config[n]!=anchor.get(n))})
        self.state['executions']=self.state['executions'][-self.settings.macro_interval:]

    def candidate_pool(self,gp,anchor,rng,seen,total=100,audit_size=4):
        modes={k:v for k,v in sorted(self.state['modes'].items())};ng=total-20
        if len(modes)*(audit_size+1)>ng:raise ValueError('Too many directions for candidate capacity')
        audits={key:self.sample(m,anchor,rng,audit_size,seen) for key,m in sorted(modes.items())}
        unavailable=[key for key,v in audits.items() if not v]
        audits={key:v for key,v in audits.items() if v}
        modes={key:mode for key,mode in modes.items() if key in audits}
        m=min(map(len,audits.values()),default=0)
        audits={key:v[:m] for key,v in audits.items()}
        gains,_,_=joint_gains(gp,anchor,list(audits.values()),rng,self.settings.posterior_samples,self.settings.delta)
        values={key:float(g.mean()) for key,g in zip(audits,gains)}
        minimum=audit_size+1;remaining=ng-minimum*len(modes)
        weights=np.array([.02+values[k] for k in modes])
        if len(weights):weights/=weights.sum()
        exact=remaining*weights;counts=np.floor(exact).astype(int)+minimum
        for i in np.argsort(-(exact-np.floor(exact)),kind='stable')[:ng-int(counts.sum())]:counts[i]+=1
        quotas=dict(zip(modes,map(int,counts)))
        pool={};sources={}
        def add(cfg,key):
            h=self.space.config_key(cfg)
            if h in seen:return False
            pool[h]=cfg;sources.setdefault(h,[]).append(key);return True
        for key,mode in modes.items():
            for cfg in audits[key]:add(cfg,key)
            current=set(pool)|set(seen)
            available=quotas[key]-len(audits[key])
            for patch in self.state['proposals'][mode['expert']].get('patches',[]):
                if available<=0:break
                if not set(patch)<=set(mode['knobs']):continue
                cfg=strict_patch(self.space,anchor,patch,mode['knobs'])
                if cfg is not None and inside_region(self.space,cfg,mode['knobs'],mode['regions'],self.unit):
                    h=self.space.config_key(cfg)
                    if h not in current and add(cfg,key):available-=1;current.add(h)
            for cfg in self.sample(mode,anchor,rng,available,current):add(cfg,key)
        # Broad active-only and cross-group references; never forced to execute.
        active=self.state['active']
        reference_count=0
        for _ in range(256):
            if len(pool)>=total or reference_count>=20:break
            count=min(len(active),int(rng.integers(1,5)))
            names=sorted(rng.choice(active,size=count,replace=False).tolist())
            names=sorted(self.closure(names))
            if not set(names)<=set(active):continue
            ref=self.make_mode(EXPERTS[0],names,{},anchor,'reference')
            for cfg in self.sample(ref,anchor,rng,1,set(pool)|set(seen),full=True):
                if add(cfg,'reference'):reference_count+=1
        if not pool:raise RuntimeError('No legal untested candidates')
        return list(pool.values()),sources,{'audit_size':m,'opportunities':values,'quotas':quotas,
                                           'pool_size':len(pool),'ng':ng,'nr':20,
                                           'reference_count':reference_count,
                                           'unavailable_directions':unavailable}
