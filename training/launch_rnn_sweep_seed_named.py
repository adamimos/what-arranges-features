"""Parallel-GPU launcher for train_rnn.py (run directories named by seed); adapted from launch_transformer_sweep.py."""

import argparse
import itertools
import subprocess
import yaml
import os
from datetime import datetime
import torch
import time


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_config_sweep(config):
    global_config = config.get('global_config', {})
    model_config = config.get('model_config', {})
    train_config = config.get('train_config', {})

    sweep_config = config.get('sweep_config', {})
    sweep_train_config = sweep_config.get('train_config', {})
    sweep_model_config = sweep_config.get('model_config', {})
    sweep_process_config = sweep_config.get('process_config', {})

    model_config_combinations = [
        dict(zip(sweep_model_config.keys(), combo))
        for combo in itertools.product(*sweep_model_config.values())
    ]
    train_config_combinations = [
        dict(zip(sweep_train_config.keys(), combo))
        for combo in itertools.product(*sweep_train_config.values())
    ]
    for cfg in model_config_combinations:
        cfg.update(model_config)
    for cfg in train_config_combinations:
        cfg.update(train_config)

    combined = itertools.product(model_config_combinations,
                                  train_config_combinations,
                                  sweep_process_config)
    return ({
        'global_config': global_config,
        'model_config': mc, 'train_config': tc, 'process_config': pc,
    } for mc, tc, pc in combined)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--runs_per_gpu', type=int, default=1)
    p.add_argument('--gpus', type=str, default=None,
                   help='Comma-separated GPU ids to use (default: all).')
    p.add_argument('--sweep_id', type=str, default=None,
                   help='Override sweep id (else timestamp). Required when '
                        'multiple launchers run concurrently to avoid '
                        'directory collisions.')
    p.add_argument('--cpu_workers', type=int, default=0,
                   help='If >0, run on CPU with this many parallel workers '
                        '(ignores --gpus). Useful when GPUs are wedged.')
    p.add_argument('--cpu_threads_per_worker', type=int, default=4,
                   help='OMP/MKL thread count per CPU worker.')
    args = p.parse_args()

    sweep_config = load_config(args.config)
    sweep_params = list(create_config_sweep(sweep_config))

    sweep_id = (args.sweep_id or sweep_config.get('sweep_id') or
                datetime.now().strftime("%Y%m%d%H%M%S"))
    sweep_dir = f"{sweep_config['global_config']['output_dir']}/{sweep_id}"
    os.makedirs(sweep_dir, exist_ok=True)
    with open(f"{sweep_dir}/sweep_config.yaml", 'w') as f:
        yaml.dump(sweep_config, f)

    cpu_mode = args.cpu_workers > 0
    if cpu_mode:
        # In CPU mode, "available_slots" is just N copies of None as a slot token.
        available_slots = [None] * args.cpu_workers
    else:
        if args.gpus is not None:
            gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
        else:
            gpu_list = list(range(torch.cuda.device_count()))
        available_slots = gpu_list * max(1, args.runs_per_gpu)
    available_gpus = available_slots  # keep variable name for backward compat
    print(f"Sweep id: {sweep_id}")
    slots_desc = (available_slots if not cpu_mode else
                  f'{args.cpu_workers} CPU workers @ {args.cpu_threads_per_worker} threads each')
    print(f"Mode: {'CPU' if cpu_mode else 'GPU'}, slots: {slots_desc}")
    print(f"Runs to launch: {len(sweep_params)}")

    queue = list(sweep_params)
    running = []
    run_counter = 0
    while queue or running:
        while queue and available_gpus:
            slot = available_gpus.pop(0)
            cfg = queue.pop(0)
            mc = cfg['model_config']
            name = (f"run_{run_counter}_L{mc['num_layers']}_H{mc['hidden_size']}"
                    f"_{mc['rnn_type']}_seed{mc.get('seed','42')}"
                    f"_{cfg['process_config']['name']}")
            edir = f"{sweep_dir}/{name}"
            os.makedirs(edir, exist_ok=True)
            cpath = f"{edir}/run_config.yaml"
            cfg['run_id'] = name
            cfg['global_config'] = dict(cfg['global_config'])
            cfg['global_config']['sweep_id'] = sweep_id
            cfg['global_config']['device'] = 'cpu' if cpu_mode else f'cuda:{slot}'
            cfg['config_path'] = cpath
            cfg['experiment_dir'] = edir
            with open(cpath, 'w') as f:
                yaml.dump(cfg, f)

            env = os.environ.copy()
            if cpu_mode:
                env['OMP_NUM_THREADS'] = str(args.cpu_threads_per_worker)
                env['MKL_NUM_THREADS'] = str(args.cpu_threads_per_worker)
                env['CUDA_VISIBLE_DEVICES'] = ''  # hide broken/wedged GPUs
                cmd = ['python', './training/train_rnn.py', '--config', cpath]
                slot_label = 'cpu'
            else:
                cmd = ['python', './training/train_rnn.py',
                       '--config', cpath, '--parallel', '--gpu_id', str(slot)]
                slot_label = f'cuda:{slot}'

            stdout_log = open(f"{edir}/stdout.txt", "w")
            stderr_log = open(f"{edir}/stderr.txt", "w")
            proc = subprocess.Popen(cmd, stdout=stdout_log, stderr=stderr_log,
                                     env=env)
            running.append({'p': proc, 'gpu': slot, 'name': name,
                            'so': stdout_log, 'se': stderr_log,
                            'slot_label': slot_label})
            run_counter += 1
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] launched "
                  f"{name} on {slot_label} (pid={proc.pid})")

        for r in running[:]:
            rc = r['p'].poll()
            if rc is not None:
                running.remove(r)
                r['so'].close(); r['se'].close()
                available_gpus.append(r['gpu'])
                status = 'OK' if rc == 0 else f'ERR(rc={rc})'
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status:<7s} "
                      f"{r['name']} on {r['slot_label']}")
        time.sleep(5)

    print(f"\nSweep complete: {sweep_dir}")


if __name__ == "__main__":
    main()
