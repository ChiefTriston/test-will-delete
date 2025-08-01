# modules/git_sync/git_sync.py
"""
Final manifest composition and GitHub sync using GitPython.
Outputs job_manifest.json and last_git_commit.json.
Pushes to GitHub.
"""

import json
import git
import os
import portalocker
import shutil
from datetime import datetime
import numpy as np
import time

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    config = context['config']
    
    jm = {'job_id':context['job_id'],'status':'complete'}
    jm['total_slices'] = sum(
      len(json.load(open(os.path.join(context['output_dir'],'emotion_tags',sp,'transcript.json')))['slices'])
      for sp in context['speaker_ids']
    )
    # compute flagged, slope, entropy
    jm['flagged_segments'] = sum(
      len([a for a in json.load(open(os.path.join(context['output_dir'],'emotion_tags',sp,'drift_vector.json')))['anomalies']])
      for sp in context['speaker_ids']
    )
    arc = json.load(open(os.path.join(context['output_dir'],'arc_classification.json')))
    jm['arc'] = arc['arc']
    jm['confidence_drift_slope'] = np.mean([
      json.load(open(os.path.join(context['output_dir'],'emotion_tags',sp,'drift_log.json')))['confidence_drift_slope']
      for sp in context['speaker_ids']
    ])
    jm['emotion_entropy'] = np.mean([
      json.load(open(os.path.join(context['output_dir'],'emotion_tags',sp,'fingerprint.json')))['entropy']
      for sp in context['speaker_ids']
    ])
    jm['observer_feedback'] = os.path.exists(os.path.join(context['output_dir'],'learned_rules.json'))

    manifest_path = os.path.join(output_dir, 'job_manifest.json')
    with open(manifest_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(jm, f)
        portalocker.unlock(f)
    
    repo_path = config['global']['github_repo_path']
    repo = git.Repo(repo_path)
    
    branch = config['git_sync'].get('branch', 'main')
    remote_name = config['git_sync'].get('remote', 'origin')
    repo.git.checkout(branch)
    
    last_commit = repo.head.commit
    commit_info = {
        'hash': last_commit.hexsha,
        'timestamp': datetime.fromtimestamp(last_commit.authored_date).isoformat(),
        'pushed_by': repo.git.config('user.name')
    }
    commit_path = os.path.join(output_dir, 'last_git_commit.json')
    with open(commit_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(commit_info, f)
        portalocker.unlock(f)
    
    # Copy output to repo/emotion_tags/{job_id}
    target_dir = os.path.join(repo_path, config['global']['github_target_dir'], 'emotion_tags', context['job_id'])
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(output_dir, target_dir, dirs_exist_ok=True)
    
    repo.git.add(A=True)
    repo.index.commit(f"Emotion tags for job {context['job_id']}")
    
    origin = repo.remote(name=remote_name)
    retries = 3
    for attempt in range(retries):
        try:
            origin.push()
            break
        except git.exc.GitCommandError as e:
            print(f"Push failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                # Rollback commit
                repo.git.reset('--hard', 'HEAD~1')
                raise
    
    return {'job_manifest': manifest_path, 'last_git_commit': commit_path}