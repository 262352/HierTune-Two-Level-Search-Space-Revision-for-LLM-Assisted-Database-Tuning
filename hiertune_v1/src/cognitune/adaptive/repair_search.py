"""Keep expert directions while retaining a GP-local incumbent candidate channel."""
import copy
import numpy as np
from .faithful_group_search import FaithfulGroupSearch
from .nested_search import strict_patch


class RepairSearch(FaithfulGroupSearch):
    local_capacity=20

    def candidate_pool(self,gp,anchor,rng,seen,total=100,audit_size=4):
        modes=sorted(self.state['modes'].items());local=[];local_sources={};excluded=set(seen)
        for attempt in range(1000):
            if len(local)>=min(self.local_capacity,total//5) or not modes:break
            key,mode=modes[attempt%len(modes)]
            numeric=[n for n in mode['knobs'] if self.space.by_name[n].kind in {'integer','float','ordinal'}]
            names=mode['knobs'] if attempt%2 else [rng.choice(numeric or mode['knobs'])]
            group=self.closure(names)
            patch={}
            for name in group:
                knob=self.space.by_name[name];value=anchor.get(name,knob.default)
                if knob.kind in {'integer','float'}:
                    width=min(.05,mode['regions'][name].get('halfwidth',.15)/4)
                    z=float(np.clip(self.unit(knob,value)+rng.normal(0,max(width,1e-6)),0,1))
                    patch[name]=self.value_at(knob,z)
                elif knob.kind=='ordinal':
                    i=knob.choices.index(value);j=int(np.clip(i+rng.choice([-1,1]),0,len(knob.choices)-1))
                    patch[name]=knob.choices[j]
                else:
                    # Hold categorical choices in mixed numeric local probes;
                    # ordinary expert/global channels still explore categories.
                    patch[name]=value
                    if not numeric:
                        choices=[False,True] if knob.kind=='bool' else list(knob.choices)
                        patch[name]=choices[int(rng.integers(len(choices)))]
            cfg=strict_patch(self.space,anchor,patch,group)
            if cfg is None:continue
            digest=self.space.config_key(cfg)
            if digest in excluded:continue
            excluded.add(digest);local.append(cfg);local_sources[digest]=['incumbent_local',key]
        regular,sources,report=super().candidate_pool(gp,anchor,rng,excluded,total-len(local),audit_size)
        configs=regular+local;sources.update(local_sources)
        report.update(capacity=total,pool_size=len(configs),incumbent_local_count=len(local),
            incumbent_local_radius='min(0.05, current halfwidth/4), normalized numeric units',
            candidate_sources=sources,
            regular_groups={k:list(m['knobs']) for k,m in self.state['modes'].items()},
            acquisition_sampler='round-shared Sobol QMC, at least 1024 draws')
        report['active_knob_candidate_exposure']={n:sum(c.get(n)!=anchor.get(n) for c in configs) for n in self.state['active']}
        report['unexposed_active_knobs']=[n for n,count in report['active_knob_candidate_exposure'].items() if not count]
        return configs,sources,report
