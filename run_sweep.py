#!/usr/bin/python

import argparse
import subprocess
import re
import os
import time
import sys
import psutil


parser = argparse.ArgumentParser()
parser.add_argument('yaml', type=str, help='the yaml file for sweep.')
parser.add_argument('--n_agent', type=int, default=None, help='the number of agents to run.')
parser.add_argument('--gpus', type=str, default=None, help='the GPUs to use, e.g.: 1,2,3')
parser.add_argument('--log_dir', type=str, default='outputs/', help='the dir for logging.')
parser.add_argument('--sweep', type=str, default=None, help='continue the existing sweep.')

args = parser.parse_args()

if not args.gpus:
	import torch
	args.gpus = list(range(torch.cuda.device_count()))
else:
	args.gpus = args.gpus.split(',')

if not args.n_agent:
	args.n_agent = len(args.gpus)

print(f'Sweep {args.yaml} by {args.n_agent} agents on GPUs: {args.gpus}.')

if not args.sweep:
	create_sweep = 'wandb sweep ' + args.yaml
else:
	create_sweep = f'wandb sweep --update {args.sweep} {args.yaml}'
result = subprocess.run(create_sweep, shell=True, capture_output=True, text=True)
print(result.stderr)
run_agent = re.search('wandb agent .*', result.stderr).group(0)
sweep_id = run_agent.split(' ')[-1]
args.log_dir = os.path.join(args.log_dir, sweep_id)
os.makedirs(args.log_dir, exist_ok=True)

active_agents = []
for agent in range(args.n_agent):
	gpu = args.gpus[agent % len(args.gpus)]
	log_file = os.path.join(args.log_dir, f'agent_{agent}.log')
	print(f'Running Agent {agent} on GPU {gpu}, logging to {log_file}.')
	p = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu} exec {run_agent} >{log_file} 2>&1', shell=True)
	active_agents.append((agent, p))


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


try:
	print('Waiting all agents to finish...')
	while active_agents:
		time.sleep(10)
		finished = []
		for i, (agent, p) in enumerate(active_agents):
			if p.poll() is not None:
				finished.append(i)
		
		if finished:
			print(f'Agent finished: {[active_agents[i][0] for i in finished]}, ', end='')
			for i in sorted(finished, reverse=True):
				del active_agents[i]
			print(f'remains: {[x[0] for x in active_agents]}.')

	print('All agents finished.')

except KeyboardInterrupt:
	print("\nKilling all active agents...")
	for agent, p in active_agents:
		kill(p.pid)
	sys.exit(0)