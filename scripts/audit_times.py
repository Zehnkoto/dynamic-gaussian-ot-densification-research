#!/usr/bin/env python3
"""Audit temporal overlap separately from the dataset's camera/view split."""
import json, os
from pathlib import Path

root = Path(os.environ['RESEARCH_ROOT'])
data = root / 'datasets/bouncingballs'
report = {'source': str(data), 'splits': {}, 'overlap': {}}
times = {}
for split in ['train', 'val', 'test']:
    frames = json.loads((data / ('transforms_' + split + '.json')).read_text())['frames']
    times[split] = {round(frame['time'], 12) for frame in frames}
    report['splits'][split] = {'frames': len(frames), 'unique_times': len(times[split]),
                               'time_range': [min(times[split]), max(times[split])]}
for split in ['val', 'test']:
    report['overlap'][split] = {'shared_with_train': len(times[split] & times['train']),
                                'unseen_from_train': sorted(times[split] - times['train'])}
report['note'] = 'Official view splits are not necessarily unseen-time splits. Define and record a separate temporal protocol.'
print(json.dumps(report, indent=2))
