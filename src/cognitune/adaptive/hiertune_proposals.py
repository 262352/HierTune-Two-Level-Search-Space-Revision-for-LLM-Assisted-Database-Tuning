"""Structured, cached specialist requests for the epoch-12 policy."""
from __future__ import annotations
import hashlib
import copy
import json
import time
from pathlib import Path
from .prompt_context import shared_history_prompt, token_usage
from .proposals import normalize_proposal, validate_proposal
from cognitune.llm.deepseek import _parse_json_object

SYSTEM = (
 "You are one of four PostgreSQL domain specialists. Return one JSON object; "
 "never invent measured performance. Use the entire supplied execution history. "
 "The controller, not you, computes posterior opportunity and accepts structural changes. "
 "Use actual legal values and only allowed_knobs. For phase bootstrap, score EVERY "
 "allowed knob from 0 to 100 in scores. Scores need not sum to 100. Otherwise omit scores. "
 "Provide at most two sparse groups, each {knobs:[names],reason:string}, "
 "normally 1 to 4 interacting knobs. For proposal phase groups must be active; "
 "for macro phase suggest both active groups and potential replacements. "
 "For macro phase provide partners:{inactive_knob:[1 to 3 active partner names]} "
 "for EVERY owned_inactive knob. Include a conditional parent when necessary. "
 "Give regions only for knobs you recommend: numeric {center:actual_value,"
 "halfwidth:0.01..0.5,action:hold,reason:string}; discrete {values:[legal_values],"
 "action:hold,reason:string}. No interpolation of categories. Give up to four "
 "patches of at most four knobs. The shape is {scores:{},groups:[],partners:{},"
 "regions:{},patches:[],joint_hypotheses:[]}. Keep explanations concise."
)

def normalize_answer(answer,context):
    """Separate group hypotheses from executable values, without inventing values.

    A known inactive group is deferred intact, never projected into a different
    interaction. Metadata-only 'patches' contain no executable configuration.
    """
    answer=copy.deepcopy(answer);repairs=[]
    known={k['name'] for k in context.get('global_knob_domains',context['allowed_knobs'])}
    allowed={k['name'] for k in context['allowed_knobs']}
    groups=answer.get('groups',[])
    if not isinstance(groups,list):raise ValueError('groups must be a list')
    groups=list(groups);patches=[];deferred_patches=[]
    for patch in answer.get('patches',[]):
        if isinstance(patch,dict) and isinstance(patch.get('knobs'),list) and set(patch)<= {'knobs','reason','mechanism'}:
            groups.append({'knobs':patch['knobs'],'reason':patch.get('reason',patch.get('mechanism',''))})
            repairs.append({'action':'group_metadata_is_not_configuration','original':patch})
        elif isinstance(patch,dict) and set(patch)<=known and not set(patch)<=allowed:
            deferred_patches.append(patch)
        else:patches.append(patch)
    answer['patches']=patches
    if deferred_patches:
        answer['deferred_patches']=deferred_patches
        repairs.append({'action':'defer_cross_scope_patch_intact','patches':deferred_patches})
    retained=[];deferred=[];seen=set()
    for group in groups:
        if not isinstance(group,dict) or not isinstance(group.get('knobs'),list):
            raise ValueError('group requires an object with a knobs list')
        names=group['knobs']
        if any(not isinstance(n,str) for n in names) or not set(names)<=known:
            raise ValueError('unknown group knob; use declared global domains')
        key=tuple(sorted(set(names)))
        if key in seen:continue
        seen.add(key)
        if not set(names)<=allowed or len(retained)>=2:
            deferred.append(group);continue
        retained.append(group)
    answer['groups']=retained
    if deferred:
        answer['deferred_groups']=deferred
        repairs.append({'action':'defer_unwritable_or_extra_groups_intact','groups':deferred})
    # Partner suggestions are advisory rather than executable values.  An LLM
    # may mention another inactive replacement candidate as a partner.  Keep
    # only the currently active context, cap it deterministically, and let the
    # controller use its domain-aware fallback when nothing usable remains.
    if context.get('phase')=='macro':
        raw_partners=answer.get('partners',{})
        if not isinstance(raw_partners,dict):
            raw_partners={}
            repairs.append({'action':'replace_non_object_partners'})
        active=set(context.get('active_knobs',[]))
        normalized={}
        for name in context.get('owned_inactive',[]):
            raw=raw_partners.get(name,[])
            if not isinstance(raw,list): raw=[]
            usable=[]
            for partner in raw:
                if isinstance(partner,str) and partner in active and partner not in usable:
                    usable.append(partner)
            normalized[name]=usable[:3]
            if raw!=normalized[name]:
                repairs.append({'action':'filter_partner_context','knob':name,
                                'original':raw,'retained':normalized[name]})
        answer['partners']=normalized
    answer,base_repairs=normalize_proposal(answer,context)
    return answer,repairs+base_repairs

class HierTuneExpertProposer:
    def __init__(self, transport, directory):
        self.transport=transport
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)

    def __call__(self, expert, context):
        user,layout=shared_history_prompt(expert,context)
        payload={'model':self.transport.model,'messages':[{'role':'system','content':getattr(self,'system',SYSTEM)},
                    {'role':'user','content':user}], 'response_format':{'type':'json_object'},
                 'thinking':{'type':'disabled'},'temperature':.2,'max_tokens':8000}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        path=self.directory/(digest+'.json')
        if path.exists():
            answer,_=normalize_answer(json.loads(path.read_text())['answer'],context)
            self.validate(answer,context)
            self.metric(expert,context,origin='local_cache',key=digest,layout=layout)
            return answer
        failed=path.with_suffix('.failed.json')
        if failed.exists():
            old=json.loads(failed.read_text())
            for index in reversed(range(len(old['attempts']))):
                try:
                    answer,_=_parse_json_object(old['attempts'][index]['content'])
                    answer,repairs=normalize_answer(answer,context)
                    self.validate(answer,context)
                except (ValueError,TypeError,KeyError):continue
                path.write_text(json.dumps({'answer':answer,'attempts':old['attempts'],
                    'repairs':repairs,'revalidated_attempt':index,'source_failed_file':failed.name},indent=2))
                self.metric(expert,context,origin='revalidated_provider_response',key=digest,
                    source_attempt=index+1,layout=layout)
                return answer
        attempts=[]
        for attempt in range(3):
            started=time.monotonic()
            response=self.transport._post(payload)
            raw=response['choices'][0]['message'].get('content') or ''
            elapsed=time.monotonic()-started
            stats=token_usage(response.get('usage'))
            self.metric(expert,context,origin='provider',key=digest,attempt=attempt+1,
                        elapsed_seconds=elapsed,model=response.get('model'),layout=layout,**stats)
            entry={'content':raw,'usage':response.get('usage'),'elapsed_seconds':elapsed}
            attempts.append(entry)
            try:
                answer,_=_parse_json_object(raw)
                answer,repairs=normalize_answer(answer,context)
                self.validate(answer,context)
            except (ValueError,TypeError,KeyError) as error:
                entry['error']=str(error)
                payload['messages']=payload['messages'][:2]+[{'role':'user','content':
                    'Regenerate the entire valid JSON. Writable names: '+json.dumps(sorted(k['name'] for k in context['allowed_knobs']))+
                    '. groups contains {knobs:[names],reason:string}; patches contains only flat '+
                    '{actual_knob_name:actual_legal_value}, never knobs/reason metadata. Correct: '+str(error)}]
                if attempt==2:
                    path.with_suffix('.failed.json').write_text(json.dumps({'attempts':attempts},indent=2))
                    raise
                continue
            path.write_text(json.dumps({'answer':answer,'attempts':attempts,'repairs':repairs},indent=2))
            return answer
        raise RuntimeError('unreachable')

    @staticmethod
    def validate(answer, context):
        validate_proposal(answer,context)
        allowed={k['name'] for k in context['allowed_knobs']}
        if context['phase']=='bootstrap':
            scores=answer.get('scores',{})
            if set(scores)!=allowed or any(type(v) not in (int,float) or not 0<=v<=100 for v in scores.values()):
                raise ValueError('scores must rate EVERY allowed knob with a finite number in [0,100]')
        groups=answer.get('groups',[])
        if not isinstance(groups,list) or len(groups)>2:
            raise ValueError('at most two groups required')
        for group in groups:
            if not isinstance(group,dict):raise ValueError('each group must be an object')
            names=group.get('knobs')
            if not isinstance(names,list) or not 1<=len(names)<=6 or not set(names)<=allowed:
                raise ValueError('groups must contain 1..6 allowed knob names')
        partners=answer.get('partners',{})
        if not isinstance(partners,dict):
            raise ValueError('partners must be an object')
        active=set(context.get('active_knobs',[]))
        if context['phase']=='macro':
            for name,group in partners.items():
                if name not in allowed or not isinstance(group,list) or len(group)>3 or not set(group)<=active:
                    raise ValueError('partner groups may use up to 3 active names')

    def metric(self, expert, context, **fields):
        item={'time':time.time(),'expert':expert,'phase':context['phase'],
              'trial':context.get('trial'),**fields}
        with (self.directory/'request_metrics.jsonl').open('a') as f:
            f.write(json.dumps(item,sort_keys=True)+'\n')
        print('[HierTune LLM] '+json.dumps(item,sort_keys=True),flush=True)
