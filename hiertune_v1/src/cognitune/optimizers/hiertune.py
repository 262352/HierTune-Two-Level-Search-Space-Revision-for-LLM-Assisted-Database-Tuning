"""Execution-feedback-driven HierTune epoch-12 optimizer, separately versioned."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cognitune.adaptive.epoch_search import EpochSearch,EpochSettings
from cognitune.adaptive.prompt_context import full_execution_history
from cognitune.agents.ownership import EXPERTS
from cognitune.experiment.recorder import RunRecorder,TrialRecord
from cognitune.experiment.resilience import atomic_json
from cognitune.optimizers.cognitune_v3 import unit,value_at
from cognitune.surrogate.gp import GPPerformanceSurrogate

VERSION='hiertune-epoch12-v1'

def run_hiertune(space,evaluator,initial_configs,budget,seed,candidate_pool_size,run_dir,*,
                 proposer,cognitive_context=None,resume=False,settings=None,active_target=18,
                 surrogate_factory=GPPerformanceSurrogate,checkpoint_callback=None,
                 controller_factory=EpochSearch,method_version=VERSION,
                 reference_observations=None,posterior_reference=False):
    if len(initial_configs)<2 or budget<len(initial_configs) or candidate_pool_size<60:
        raise ValueError('Invalid initial design, budget or candidate capacity')
    directory=Path(run_dir);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'trials.jsonl').exists() and not resume:raise FileExistsError('Trace already exists')
    recorder=RunRecorder(directory,resume=resume)
    settings=settings or EpochSettings()
    ctrl=controller_factory(space,unit,value_at,settings=settings,active_target=active_target)
    rng=np.random.default_rng(seed)
    controls=reference_observations or []
    params={'version':method_version,'space':space.fingerprint(),'budget':budget,'seed':seed,
      'candidate_pool_size':candidate_pool_size,'active_target':active_target,'settings':settings.__dict__,
      'initial_sha256':hashlib.sha256(json.dumps(initial_configs,sort_keys=True).encode()).hexdigest(),
      'context_sha256':hashlib.sha256(json.dumps(cognitive_context or {},sort_keys=True).encode()).hexdigest()}
    params.update(posterior_reference=posterior_reference,
        reference_observations_sha256=hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest())
    checkpoint_path=directory/'epoch_checkpoint.json'
    pending_path=directory/'pending_proposal.json'
    diag_path=directory/'round_diagnostics.jsonl'

    def event(kind,**fields):
        row={'event':kind,'time':time.time(),**fields}
        with diag_path.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
        # Detailed centers, opportunities, quotas and proposal comparisons stay inspectable.
        print('[HierTune] '+json.dumps(row,sort_keys=True),flush=True)

    def save():
        atomic_json(checkpoint_path,{'parameters':params,'completed':len(recorder.records),
                    'controller':ctrl.state,'rng':rng.bit_generator.state})

    def apply_plan(plan,result):
        if plan['adaptive']:
            if hasattr(ctrl,'observe_result'):
                ctrl.observe_result(plan,result)
            else:ctrl.observe(plan['anchor'],plan['config'],result.status,plan['trial'])

    if checkpoint_path.exists():
        old=json.loads(checkpoint_path.read_text())
        if old['parameters']!=params:raise ValueError('Checkpoint protocol differs')
        ctrl.state=old['controller'];rng.bit_generator.state=old['rng']
        if old['completed']!=len(recorder.records):
            if old['completed']+1!=len(recorder.records) or not pending_path.exists():
                raise ValueError('Checkpoint/trace mismatch without a recoverable pending trial')
            plan=json.loads(pending_path.read_text())
            last=recorder.records[-1]
            if plan['trial']!=last.trial or plan['config']!=last.config:
                raise ValueError('Pending result does not match trace')
            ctrl.state=plan['controller'];rng.bit_generator.state=plan['rng']
            apply_plan(plan,last);save();pending_path.unlink()
    elif recorder.records:
        raise ValueError('Cannot resume without epoch checkpoint')
    else:save()
    if pending_path.exists():
        pending=json.loads(pending_path.read_text())
        if pending['trial']==len(recorder.records)-1 and recorder.records[-1].config==pending['config']:
            pending_path.unlink()  # completed checkpoint, interrupted before pending cleanup

    domains=[{**k.__dict__,'condition':str(k.condition)} for k in space.knobs]
    by_name={k['name']:k for k in domains}

    def ask(phase,anchor,trial,experts=EXPERTS):
        history=full_execution_history(recorder.records)
        for row,record in zip(history,recorder.records):
            metadata=record.metadata or {}
            row['measured_success_fraction']=metadata.get('measured_success_fraction')
            row['benchmark_error_evidence']=metadata.get('benchmark_error_evidence',{})
        active=ctrl.state['active'];started=time.monotonic()
        def request(expert):
            owned=sorted(n for n in space.by_name if expert in ctrl.ownership[n])
            allowed=owned if phase=='bootstrap' else (active if phase in {'proposal','regroup'} else sorted(space.by_name))
            ctx={'phase':phase,'trial':trial,'remaining_trials':budget-trial,
                'workload':cognitive_context or {},'global_knob_domains':domains,
                'allowed_knobs':[by_name[n] for n in allowed],
                'active_knobs':active,'owned_knobs':owned,
                'owned_inactive':[n for n in owned if n not in active] if phase=='macro' else [],
                'anchor':anchor,'feedback':history,'candidate_quota':20,
                'current_modes':[m for m in ctrl.state['modes'].values() if m['expert']==expert],
                'instruction':'Propose sparse interacting knob groups. Geometry and adoption are computed by the GP controller. '
                   'A low observed gain alone does not establish that a knob is intrinsically unimportant.'}
            if hasattr(ctrl,'expert_context'):ctx.update(ctrl.expert_context(expert,phase))
            return proposer(expert,ctx)
        with ThreadPoolExecutor(max_workers=4) as executor:
            answers=dict(zip(experts,executor.map(request,experts)))
        event('expert_refresh',phase=phase,trial=trial,history_count=len(history),
              elapsed_seconds=time.monotonic()-started,parallel_workers=4)
        return answers

    while len(recorder.records)<budget:
        trial=len(recorder.records)
        # A pending proposal is replayed exactly after a process/API/database failure.
        if pending_path.exists():
            plan=json.loads(pending_path.read_text())
            if plan['trial']!=trial:raise ValueError('Stale pending proposal')
            ctrl.state=plan['controller'];rng.bit_generator.state=plan['rng']
            event('pending_replay',trial=trial)
        elif trial<len(initial_configs):
            plan={'trial':trial,'config':initial_configs[trial],'anchor':space.get_default(),
                  'adaptive':False,'source':'shared_initial','state':{'initial':True},
                  'controller':copy.deepcopy(ctrl.state),'rng':rng.bit_generator.state,
                  'compute_seconds':0.,'acquisition':None,'mean':None,'std':None}
        else:
            started=time.monotonic()
            valid=[r for r in recorder.records if r.status=='SUCCESS']
            if len(valid)<2:raise RuntimeError('At least two real successful observations required')
            anchor_record=max(valid,key=lambda r:r.utility);anchor=copy.deepcopy(anchor_record.config)
            seen={space.config_key(r.config) for r in recorder.records}
            gp=surrogate_factory(space);fit_start=time.monotonic()
            gp.fit([r.config for r in valid]+[r['config'] for r in controls],
                   [r.utility for r in valid]+[r['utility'] for r in controls])
            if hasattr(gp,'begin_round'):gp.begin_round(seed+trial)
            reference_source='observed_best'
            if posterior_reference:
                references=list(valid)
                if controls:
                    default=dict(controls[0]);default['utility']=float(np.median([r['utility'] for r in controls]))
                    default['raw_metric']=float(np.median([r['raw_metric'] for r in controls]))
                    default['trial']=None
                    references.append(SimpleNamespace(**default))
                ref_prediction=gp.posterior([r.config for r in references])
                anchor_record=references[int(np.argmax(ref_prediction.mean))]
                anchor=copy.deepcopy(anchor_record.config)
                reference_source='measured_default' if anchor_record.trial is None else 'posterior_incumbent'
            # A measured default is already executed even though outside the
            # 50 tuning observations; do not silently re-execute it as a trial.
            seen.update(space.config_key(r['config']) for r in controls)
            fit_seconds=time.monotonic()-fit_start
            if not ctrl.state['active']:
                boot=ask('bootstrap',anchor,trial)
                ctrl.initialize(boot,anchor)
                proposals=ask('proposal',anchor,trial)
                ctrl.state['proposals']=proposals
                ctrl.state['modes']=ctrl.build_modes(ctrl.state['active'],proposals,anchor,ctrl.state['modes'])
                event('initialize',trial=trial,active=ctrl.state['active'],scores=ctrl.state['scores'],
                      modes=ctrl.state['modes'])
            adaptive_completed=trial-len(initial_configs)
            if hasattr(ctrl,'set_round_context'):
                ctrl.set_round_context(adaptive_completed,budget-len(initial_configs))
            macro=None
            if ctrl.review_due(adaptive_completed):
                answers=ask('macro',anchor,trial)
                macro=ctrl.macro(gp,anchor,rng,seen,adaptive_completed,answers)
                event('macro_review',trial=trial,**macro)
                if macro['accepted']:
                    # All experts see the adopted active context, before proposing new combinations.
                    answers=ask('proposal',anchor,trial)
                    ctrl.state['proposals']=answers
                    # A newly returned group is only a proposal; keep audited groups until micro comparison.
            structural_start=time.monotonic()
            if hasattr(ctrl,'regroup_requests'):
                due=ctrl.regroup_requests(gp,anchor,rng,seen)
                if due:
                    answers=ask('regroup',anchor,trial,tuple(sorted(due)))
                    ctrl.state['proposals'].update(answers)
                    event('regroup_requested',trial=trial,experts=sorted(due),evidence=ctrl.state['regroup_requests'])
            micro=ctrl.micro(gp,anchor,rng,seen)
            configs,sources,allocation=ctrl.candidate_pool(gp,anchor,rng,seen,candidate_pool_size)
            structural_seconds=time.monotonic()-structural_start
            acq_start=time.monotonic()
            scores=np.asarray(gp.log_noisy_expected_improvement(configs,seed=seed+trial))
            if not np.isfinite(scores).all():raise ValueError('Non-finite qLogNEI')
            chosen=int(np.argmax(scores));config=configs[chosen]
            candidate_prediction=gp.posterior(configs)
            pred=SimpleNamespace(mean=[candidate_prediction.mean[chosen]],std=[candidate_prediction.std[chosen]])
            audit=[{'config':c,'config_key':space.config_key(c),'sources':sources[space.config_key(c)],
                    'qlognei':float(scores[i]),'mean':float(candidate_prediction.mean[i]),
                    'std':float(candidate_prediction.std[i]),'selected':i==chosen} for i,c in enumerate(configs)]
            atomic_json(directory/'candidate_tables'/f'trial_{trial:04d}.json',audit)
            state={'anchor_trial':anchor_record.trial,'reference_source':reference_source,
                   'changed_knobs':sorted(n for n in config if config[n]!=anchor.get(n)),
                   'active':ctrl.state['active'],'allocation':allocation,
                   'micro_accepted':sum(r.get('accepted',False) for r in micro.values()),
                   'macro_accepted':bool(macro and macro['accepted']),
                   'timing':{'gp_fit_seconds':fit_seconds,'structure_and_pool_seconds':structural_seconds,
                             'acquisition_seconds':time.monotonic()-acq_start}}
            event('proposal',trial=trial,anchor_trial=anchor_record.trial,active=ctrl.state['active'],
                  reference_source=reference_source,reference_config=anchor,
                  candidate_table=f'candidate_tables/trial_{trial:04d}.json',
                  micro=micro,macro=macro,allocation=allocation,timing=state['timing'],
                  selected_config=config,selected_source=sources[space.config_key(config)],
                  proposal_seconds=time.monotonic()-started)
            plan={'trial':trial,'config':config,'anchor':anchor,'adaptive':True,'source':'qLogNEI',
                  'state':state,'controller':copy.deepcopy(ctrl.state),'rng':rng.bit_generator.state,
                  'compute_seconds':time.monotonic()-started,'acquisition':float(scores[chosen]),
                  'mean':float(pred.mean[0]),'std':float(pred.std[0])}
            if hasattr(ctrl,'observe_result'):
                plan['sources']=sources[space.config_key(config)]
                plan['anchor_utility']=anchor_record.utility
        atomic_json(pending_path,plan)
        result=evaluator.evaluate(plan['config'],trial)
        successful=[r.utility for r in recorder.records if r.status=='SUCCESS']
        if result.status=='SUCCESS':successful.append(result.utility)
        best=max(successful,default=-10.)
        state=copy.deepcopy(plan['state']);state['method_compute_seconds']=plan['compute_seconds']
        recorder.append(TrialRecord(trial=trial,method='hiertune',seed=seed,config=plan['config'],
           raw_metric=result.raw_metric,utility=result.utility,best_utility=best,best_so_far=best,
           status=result.status,candidate_source=plan['source'],elapsed_seconds=result.elapsed_seconds,
           acquisition_value=plan['acquisition'],posterior_mean=plan['mean'],posterior_std=plan['std'],
           state=state,metadata=result.metadata))
        apply_plan(plan,result);save();pending_path.unlink()
        if checkpoint_callback:checkpoint_callback(directory,len(recorder.records))
        event('outcome',trial=trial,status=result.status,raw_metric=result.raw_metric,best_improvement=best,
              changed_knobs=state.get('changed_knobs',[]))
    summary=recorder.finalize()
    summary.update(method_version=method_version,parameters=params,completed=len(recorder.records))
    atomic_json(directory/'summary.json',summary)
    return summary
