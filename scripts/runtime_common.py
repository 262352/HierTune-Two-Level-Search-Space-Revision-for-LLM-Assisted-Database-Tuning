from pathlib import Path
import fcntl, gzip, hashlib, json, shutil
from cognitune.experiment.resilience import atomic_json, InfrastructureUnavailable
ROOT=Path(__file__).resolve().parents[1]
SPACE=ROOT/'configs/knobs/postgresql_60_diagnostic.yaml'

def read(path):
    return json.loads(path.read_text())

def compact(folder):
    """Lossless raw CSV/log compression; never touch another experiment."""
    previous=Path(folder)/'compressed_csv.json'
    manifest=read(previous) if previous.exists() else {}
    paths=list(Path(folder).rglob('*.csv'))+list(Path(folder).rglob('benchbase.log'))
    for path in paths:
        if path.stat().st_size<1_000_000: continue
        digest=hashlib.sha256()
        target=path.with_suffix(path.suffix+'.gz')
        with path.open('rb') as src,gzip.open(target,'wb',compresslevel=3) as dst:
            while chunk:=src.read(1024*1024):
                digest.update(chunk);dst.write(chunk)
        manifest[str(path.relative_to(folder))]={'sha256':digest.hexdigest(),'gzip':str(target.relative_to(folder))}
        path.unlink()
    if manifest: atomic_json(Path(folder)/'compressed_csv.json',manifest)

def error_evidence(folder):
    """Observed SQL failures, not inferred performance labels or knob rankings."""
    paths=list(Path(folder).glob('**/benchbase.log'))+list(Path(folder).glob('**/benchbase.log.gz'))
    evidence={}
    for path in paths:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',errors='replace') as stream:
            text=stream.read(2_000_000)
        if 'out of shared memory' in text:
            evidence['out_of_shared_memory']=True
        if 'max_pred_locks_per_transaction' in text:
            evidence['server_hint']='You might need to increase max_pred_locks_per_transaction'
        if 'sql state [53200]' in text:evidence['sqlstate']='53200'
    return evidence

def disk_check():
    if shutil.disk_usage(ROOT).free<3*1024**3:
        raise InfrastructureUnavailable('Less than 3 GiB free; retain checkpoints and pause')

def lock():
    (ROOT/'.runtime').mkdir(parents=True,exist_ok=True)
    stream=(ROOT/'.runtime/real-comparison.lock').open('a')
    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return stream
