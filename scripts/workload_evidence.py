"""Non-secret workload facts for expert prompts; no database/API execution."""
import hashlib
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def ycsb_evidence(root):
    root=Path(root)
    path=root/'.runtime/benchbase/target/benchbase-postgres/config/postgres/sample_ycsb_config.xml'
    xml=ET.parse(path).getroot()
    names=[e.findtext('name') for e in xml.findall('transactiontypes/transactiontype')]
    blocks=[]
    for e in xml.findall('works/work'):
        weights=[float(x) for x in e.findtext('weights').split(',')]
        if len(weights)!=len(names) or sum(weights)<=0:raise ValueError('Incompatible YCSB transaction weights')
        blocks.append({n:w/sum(weights) for n,w in zip(names,weights)})
    source=root/'.runtime/benchbase/src/main/java/com/oltpbenchmark/benchmarks/ycsb/procedures'
    declarations={};hashes={}
    for name in names:
        p=source/(name+'.java');text=p.read_text()
        declarations[name]=[re.sub(r'\s+',' ',m.strip()) for m in re.findall(r'new SQLStmt\((.*?)\);',text,re.S)]
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return {'transaction_mixtures':blocks,'sql_source_declarations':declarations,
        'sql_evidence_kind':'Java SQLStmt declarations, not runtime execution plans',
        'source_sha256':hashes,'template_xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'effective_rate':'unlimited','effective_isolation':'TRANSACTION_SERIALIZABLE',
        'hardware':{'logical_cpus':os.cpu_count(),'allowed_cpus':len(os.sched_getaffinity(0)),
                    'host_memory':mem['MemTotal']}}
