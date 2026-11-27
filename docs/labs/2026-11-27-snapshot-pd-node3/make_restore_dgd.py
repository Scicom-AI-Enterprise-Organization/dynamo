import yaml, copy
d = yaml.safe_load(open('snapfix-lab/snapfix-pd-dgd.yaml'))
for c in d['spec']['components']:
    if c['name'] == 'Frontend':
        continue
    role = c['podTemplate']['metadata']['labels']['snapfix-pd/role']
    md = c['podTemplate']['metadata']
    md['labels'].pop('nvidia.com/snapshot-is-checkpoint-source', None)
    md['annotations']['nvidia.com/restore-from'] = 'snapfix-pd-' + role
    # a restore target is not a checkpoint source
    md['annotations'].pop('nvidia.com/snapshot-multi-gpu', None)
open('snapfix-lab/snapfix-pd-restore.yaml','w').write(yaml.safe_dump(d, sort_keys=False, width=200))
print('written restore dgd')
