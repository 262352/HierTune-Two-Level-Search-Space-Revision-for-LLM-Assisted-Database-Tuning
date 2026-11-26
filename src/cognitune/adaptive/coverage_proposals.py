"""Paper-specific manager and specialist protocol, caching complete history."""
from __future__ import annotations
import hashlib
import json
import math
import time
from .hiertune_proposals import HierTuneExpertProposer,SYSTEM
from cognitune.agents.ownership import EXPERTS,build_ownership
from cognitune.experiment.resilience import atomic_json
from cognitune.llm.deepseek import _parse_json_object


class CoverageProposer(HierTuneExpertProposer):
    system=SYSTEM+(
        ' This is the HierTune coverage-60 protocol. Each group has at most FOUR knobs, including dependency parents. '
        'The phase regroup requests new active-only combinations for locally stagnant spaces; propose at most two. '
        'For macro phase, also score EVERY owned_inactive knob in scores from 1 to 100 using the latest feedback. '
        'Scores at bootstrap must also be in [1,100]. Recommend regions for active knobs in your domain and '
        'high-ranking inactive entrants. Do not treat unselected rounds as local executions. '
        'The controller alone accepts region/group changes and computes NEI/coverage quotas.'
    )

    @staticmethod
    def validate(answer,context):
        HierTuneExpertProposer.validate(answer,context)
        for g in answer.get('groups',[]):
            if len(set(g['knobs']))>4:raise ValueError('A sparse group may contain at most four distinct knobs')
        if context['phase'] in {'bootstrap','macro'}:
            names=({k['name'] for k in context['allowed_knobs']} if context['phase']=='bootstrap'
                   else set(context.get('owned_inactive',[])))
            scores=answer.get('scores',{})
            if not names<=set(scores) or any(type(scores[n]) not in (float,int) or
                    not math.isfinite(scores[n]) or not 1<=scores[n]<=100 for n in names):
                raise ValueError('Provide all requested importance scores in [1,100]')

    def manager(self,space,context):
        domains=[{**k.__dict__,'condition':str(k.condition)} for k in space.knobs]
        user={'workload':context,'knobs':domains,'experts':list(EXPERTS),
              'domain_reference':build_ownership(space)}
        payload={'model':self.transport.model,'temperature':.2,'max_tokens':8000,
                 'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},
                 'messages':[{'role':'system','content':
                  'You are the HierTune PostgreSQL expert manager. For EVERY supplied knob choose relevant domain experts '
                  'and assign nonnegative importance weights summing to one. Use hardware/workload and domain relationships. '
                  'Return one JSON object: {"weights":{"knob":{"expert_name":weight}}}. Only supplied names; do not report performance.'},
                  {'role':'user','content':json.dumps(user,sort_keys=True)}]}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        path=self.directory/('manager_'+digest+'.json')
        if path.exists():
            answer=json.loads(path.read_text())['answer'];self.metric('manager',{'phase':'manager'},origin='local_cache',key=digest)
        else:
            attempts=[]
            for attempt in range(3):
                started=time.monotonic();response=self.transport._post(payload)
                raw=response['choices'][0]['message'].get('content') or ''
                attempts.append({'content':raw,'usage':response.get('usage'),'model':response.get('model')})
                self.metric('manager',{'phase':'manager'},origin='provider',key=digest,
                            elapsed_seconds=time.monotonic()-started,model=response.get('model'),usage=response.get('usage'))
                try:
                    answer,_=_parse_json_object(raw);self.check_weights(answer,space)
                except (ValueError,TypeError,KeyError) as error:
                    attempts[-1]['error']=str(error)
                    if attempt==2:
                        atomic_json(path.with_suffix('.failed.json'),{'attempts':attempts});raise
                    payload['messages']=payload['messages'][:2]+[{'role':'user','content':'Correct complete weights JSON: '+str(error)}]
                    continue
                atomic_json(path,{'answer':answer,'attempts':attempts});break
        return self.check_weights(answer,space)

    @staticmethod
    def check_weights(answer,space):
        weights=answer['weights']
        if set(weights)!=set(space.by_name):raise ValueError('Manager must assign every candidate knob')
        normalized={}
        for name,row in weights.items():
            if not isinstance(row,dict) or not row or not set(row)<=set(EXPERTS):raise ValueError('Unknown expert weight')
            if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in row.values()):raise ValueError('Invalid manager weight')
            total=sum(row.values())
            if total<=0:raise ValueError('Zero manager weight')
            normalized[name]={e:float(v/total) for e,v in row.items() if v>0}
        return normalized
