import yaml, copy, sys

src = yaml.safe_load(open('snapfix-lab/prod-fe-dgd.yaml'))
d = {'apiVersion': 'nvidia.com/v1beta1', 'kind': 'DynamoGraphDeployment',
     'metadata': {'name': 'snapfix-pd', 'namespace': 'snapfix-lab'}, 'spec': {}}
d['spec']['backendFramework'] = src['spec']['backendFramework']
d['spec']['env'] = src['spec'].get('env', [])

NODE = 'tm-ucc-c3ssfjix-0000'
MODEL_PATH = '/data/hf/swap/gemma4-31b-fp8/gemma-a'
MODEL_ID = 'Scicom-intl/gemma-4-31B-it-dpo-repetition-v3.3-handover-v20a-2026-10-27-NVFP4-FP8-attn'
RUNTIME_IMG = '865626945255.dkr.ecr.ap-southeast-5.amazonaws.com/scicom/dynamo-vllm-runtime:v1.6.0-nightly-20261016-gemma4-hetero-tp.1-cu130'
PLACEHOLDER_IMG = 'localhost/snapfix/dynamo-vllm-placeholder:v1.6.0-cu130'
AGENT_IMG = 'nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.4.0'
GPU_PREFILL = 'GPU-8a81b39c-ff3e-f36e-06e4-927977a0e1f7,GPU-979b4fa4-02a3-e0c5-726d-20ec73196cc5,GPU-9f664db9-bdac-5fdc-0bea-1dffdd229e08,GPU-23945d4b-4a48-4ea7-1551-4d10487f294e'
GPU_DECODE  = 'GPU-ef862efe-db18-8a9b-1b54-b64aa4a097db,GPU-576801ce-a8ee-703f-d940-8df565142c7b,GPU-63832da7-df65-adbd-c1cf-3d82188f3a99,GPU-0b09f3ae-9def-efd5-48d2-f45c25fbd4ea'

def envset(ct, name, value):
    for e in ct.setdefault('env', []):
        if e['name'] == name:
            e['value'] = value
            return
    ct['env'].append({'name': name, 'value': value})

def envgone(ct, name):
    ct['env'] = [e for e in ct.get('env', []) if e['name'] != name]

def argset(args, flag, value):
    for i, a in enumerate(args):
        if a == flag:
            args[i+1] = value
            return
    args += [flag, value]

def worker(c, role, gpus, ports, side, metrics, kv_events, ckpt_id):
    c['replicas'] = 1
    pt = c['podTemplate']
    md = pt.setdefault('metadata', {})
    md.setdefault('annotations', {})
    md['annotations']['nvidia.com/use-gpuuuid'] = gpus
    md['annotations']['nvidia.com/snapshot-multi-gpu'] = 'enabled'
    md['annotations']['nvidia.com/snapshot-target-containers'] = 'main'
    md.setdefault('labels', {})
    md['labels']['nvidia.com/snapshot-checkpoint-id'] = ckpt_id
    md['labels']['nvidia.com/snapshot-is-checkpoint-source'] = 'true'
    md['labels']['snapfix-pd/role'] = role
    md['labels'].pop('gemma4-31b-dpo-nvfp4fp8attn-p4d8/role', None)
    sp = pt['spec']
    sp['nodeSelector'] = {'kubernetes.io/hostname': NODE}
    sp['securityContext'] = {'seccompProfile': {'type': 'Localhost', 'localhostProfile': 'profiles/block-iouring.json'}}
    sp['hostNetwork'] = True
    sp['dnsPolicy'] = 'ClusterFirstWithHostNet'
    # drop the HF fetch init container; the model is local
    sp['initContainers'] = [i for i in sp.get('initContainers', []) if i.get('name') != 'fetch-model']
    ct = sp['containers'][0]
    ct['image'] = PLACEHOLDER_IMG
    # cuda-checkpoint --launch-job is the 1.4.0 multi-GPU contract (cuinterpose was 0.3.0)
    ct['command'] = ['/usr/local/sbin/cuda-checkpoint', '--launch-job', 'python3', '-m', 'dynamo.vllm']
    args = ct['args']
    argset(args, '--model', MODEL_PATH)
    argset(args, '--kv-cache-dtype', 'auto')
    argset(args, '--gpu-memory-utilization', '0.06')
    args += ['--served-model-name', MODEL_ID]
    argset(args, '--max-model-len', '4096')
    envset(ct, 'MODEL', MODEL_ID)
    envset(ct, 'SNAPSHOT_CONTROL_DIR', '/snapshot-control')
    envset(ct, 'DYN_SNAPSHOT_CONTROL_DIR', '/snapshot-control')
    envset(ct, 'DYN_SYSTEM_PORT', str(ports[0]))
    envset(ct, 'DYN_FORWARDPASS_METRIC_PORT', str(metrics))
    envset(ct, 'VLLM_NIXL_SIDE_CHANNEL_PORT', str(side))
    for p, name in ((ports[0], 'system'), (ports[1], 'nixl')):
        for cp in ct['ports']:
            if cp['name'] == name:
                cp['containerPort'] = p
    # kv-events endpoint
    for i, a in enumerate(args):
        if a == '--kv-events-config':
            args[i+1] = '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%d","enable_kv_cache_events":true}' % kv_events
    ct['resources']['limits']['nvidia.com/gpumem-percentage'] = '6'
    ct.setdefault('volumeMounts', [])
    ct['volumeMounts'] = [m for m in ct['volumeMounts'] if m['name'] not in ('snapshot-control','cuda-compat')]
    ct['volumeMounts'] += [
        {'name': 'snapshot-control', 'mountPath': '/snapshot-control', 'subPath': 'main'},
        {'name': 'cuda-compat', 'mountPath': '/cuda-compat', 'readOnly': True},
        {'name': 'snapshot-cuda', 'mountPath': '/tmp/snapshot-cuda', 'readOnly': True},
    ]
    sp.setdefault('volumes', [])
    sp['volumes'] = [v for v in sp['volumes'] if v['name'] not in ('snapshot-control','cuda-compat')]
    sp['volumes'] += [
        {'name': 'snapshot-control', 'emptyDir': {}},
        {'name': 'cuda-compat', 'hostPath': {'path': '/data/snapfix/cuda-compat', 'type': 'Directory'}},
        {'name': 'snapshot-cuda', 'emptyDir': {}},
    ]
    sp['initContainers'] = [i for i in sp.get('initContainers', []) if i.get('name') != 'snapshot-cuda-install']
    sp['initContainers'].append({
        'name': 'snapshot-cuda-install',
        'image': 'ghcr.io/ai-dynamo/snapshot/agent:v0.3.0',
        'imagePullPolicy': 'IfNotPresent',
        'command': ['/bin/cp'],
        'args': ['--preserve=mode', '--',
                 '/usr/local/lib/snapshot/libcuinterpose.so',
                 '/usr/local/lib/snapshot/libcuinterpose_core.so',
                 '/usr/local/bin/cuinterpose-launch',
                 '/tmp/snapshot-cuda/'],
        'volumeMounts': [{'name': 'snapshot-cuda', 'mountPath': '/tmp/snapshot-cuda'}],
    })
    return c

comps = {c['name']: c for c in src['spec']['components']}
fe = copy.deepcopy(comps['Frontend'])
fe['replicas'] = 1
fe['podTemplate']['spec']['nodeSelector'] = {'kubernetes.io/hostname': NODE}
fe['podTemplate']['spec']['containers'][0]['image'] = RUNTIME_IMG
fe['podTemplate']['spec'].pop('topologySpreadConstraints', None)
fe['podTemplate']['metadata'].pop('annotations', None)

pre = worker(copy.deepcopy(comps['VllmPrefillWorker']), 'prefill', GPU_PREFILL,
             (9090, 19090), 5600, 20380, 20080, 'snapfix-pd-prefill')
dec = worker(copy.deepcopy(comps['VllmDecodeWorker']), 'decode', GPU_DECODE,
             (9091, 19091), 5700, 20381, 20081, 'snapfix-pd-decode')

d['spec']['components'] = [fe, pre, dec]
open('snapfix-lab/snapfix-pd-dgd.yaml','w').write(yaml.safe_dump(d, sort_keys=False, width=200))
print('written')
