#!/usr/bin/env python3
"""Run the HierTune_v2 per-round expert reference policy, or --smoke without DB/API.

Algorithm imports resolve exclusively to ./src. Real DB infrastructure
is deliberately reused from the parent repository, not duplicated or rewritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(HERE/'src'),str(ROOT/'scripts')]


def smoke():
    import numpy as np
    import torch
    from cognitune.config import load_search_space
    from cognitune.adaptive.proposals import StructuredMockProposer
    from cognitune.evaluator import SyntheticEvaluator
    from cognitune.search_space.initial_design import build_initial_design
    from cognitune.optimizers.cognitune_v3 import run_cognitune_v3
    import cognitune.optimizers.cognitune_v3 as optimizer
    torch.set_num_threads(2)
    space=load_search_space(HERE/'configs/knobs/synthetic.yaml')
    calls=[];mock=StructuredMockProposer()
    def proposer(expert,context):
        calls.append({'expert':expert,'trial':context['trial'],'phase':context.get('phase','candidate_proposal')})
        return mock(expert,context)
    with tempfile.TemporaryDirectory() as directory:
        result=run_cognitune_v3(space,SyntheticEvaluator(space,0,'throughput',1000,0),
            build_initial_design(space,0,3)['configs'],5,0,60,Path(directory),proposer=proposer,
            search_policy='stable',active_target=4,audit_size=2,posterior_samples=32,fantasies=4)
    for trial in (3,4):
        assert len([c for c in calls if c['trial']==trial and c['phase']=='candidate_proposal'])==4
    assert Path(optimizer.__file__).resolve().is_relative_to(HERE)
    print(json.dumps({'status':'PASS','algorithm_file':optimizer.__file__,'trials':result['trials'],
        'calls':calls,'database_calls':0,'provider_calls':0},indent=2))


def run(args):
    import numpy as np
    import torch
    import real_benchbase as h
    import runtime_common as shared
    from baseline_workloads import WorkloadBackend,SPECS
    from build_baseline_benchbase import build
    from cognitune.config import load_search_space
    from cognitune.experiment.resilience import DurableEvaluator,atomic_json
    from cognitune.adaptive.proposals import FeedbackExpertProposer
    from cognitune.llm.deepseek import DeepSeekLLMClient
    from cognitune.optimizers.cognitune_v3 import run_cognitune_v3
    from cognitune.agents.ownership import EXPERTS
    args.output.mkdir(parents=True,exist_ok=True)
    with shared.lock():
        shared.disk_check();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2)
        space=load_search_space(HERE/'configs/knobs/postgresql_60_diagnostic.yaml')
        source=ROOT/'artifacts/ours_ycsb_ch_100_dev_v1/ycsb/preparation.json'
        initial=json.loads(source.read_text())['shared_initial']
        design=[initial[str(i)]['config'] for i in range(10)]
        backend=WorkloadBackend('ycsb',space,args.output/'measurements',build());backend.prepare()
        class CompactBackend:
            def measure(self,config,token):
                shared.disk_check();r=backend.measure(config,token);shared.compact(backend.output/token);return r
            def recover(self,token):
                backend.reset();return self.measure(space.get_default(),token)
        wrapped=CompactBackend()
        controls=DurableEvaluator(wrapped,args.output/'default_ledger')
        defaults=[controls.evaluate(space.get_default(),i).as_dict() for i in range(3)]
        if any(r['status']!='SUCCESS' for r in defaults):raise RuntimeError('Default preflight failed')
        values=[r['raw_metric'] for r in defaults];base=statistics.median(values)
        cv=statistics.stdev(values)/statistics.mean(values)
        atomic_json(args.output/'default.json',{'values':values,'median':base,'cv':cv})
        if cv>.15:raise RuntimeError('Default CV exceeds 15%; preserve controls and stop before paid API calls')
        backend.baseline={'raw_metric':base}
        controls=DurableEvaluator(wrapped,args.output/'initial_ledger')
        starts={i:controls.evaluate(c,i).as_dict() for i,c in enumerate(design)}
        if any(r['status']!='SUCCESS' for r in starts.values()):raise RuntimeError('Initial preflight failed')
        manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
        for rel,digest in manifest['source_files'].items():
            if hashlib.sha256((HERE/rel).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Archived source changed: '+rel)
        context={'dbms':'PostgreSQL 14.9','workload':'ycsb',**SPECS['ycsb'],
                 'warmup_seconds':15,'measurement_seconds':60,'normalization_default_tps':base,
                 'objective_description':'Maximize successful request throughput'}
        protocol={'name':'JointTune per-round archived stable policy','git_commit':manifest['git_commit'],
            'budget':args.budget,'initial':10,'initial_source':str(source),
            'initialization':'persisted IID configs, all freshly measured','default':base,
            'utility':'log(TPS/default), original legacy convention; plot improvement as TPS/default-1',
            'expert_schedule':'four sequential expert requests every adaptive round; bootstrap also calls experts',
            'model':'deepseek-flash','active_target':18,'candidate_capacity':100,
            'exact_historical_success_source_match':False}
        atomic_json(args.output/'protocol.json',protocol)
        transport=DeepSeekLLMClient(api_key=h.private_env()['DEEPSEEK_API_KEY'],model='deepseek-flash',max_tokens=16000)
        actual=FeedbackExpertProposer(transport,args.output/'llm_cache')
        def proposer(expert,ctx):
            with (args.output/'expert_call_schedule.jsonl').open('a') as f:
                f.write(json.dumps({'time':time.time(),'expert':expert,'trial':ctx['trial'],
                    'phase':ctx.get('phase','candidate_proposal')})+'\n')
            return actual(expert,ctx)
        def checkpoint(folder,completed):
            atomic_json(args.output/'progress.json',{'completed':completed,'target':args.budget,'time':time.time()})
        evaluator=DurableEvaluator(wrapped,args.output/'ledger',initial=starts)
        summary=run_cognitune_v3(space,evaluator,design,args.budget,0,100,args.output/'run',proposer=proposer,
            cognitive_context=context,resume=(args.output/'run/trials.jsonl').exists(),search_policy='stable',
            active_target=18,audit_size=4,eta=.8,probe_interval=8,checkpoint_callback=checkpoint)
        atomic_json(args.output/'summary.json',summary);backend.reset()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--budget',type=int,default=50)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/hiertune_v2_ycsb50')
    args=parser.parse_args()
    if args.smoke:smoke()
    else:run(args)
