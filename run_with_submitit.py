# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as classification
import submitit
import time

def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for DeiT", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/tmp/job/").is_dir():
        p = Path(f"/tmp/job/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = "/tmp/job" #get_shared_folder() / "%j"


    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=8 * num_gpus_per_node, # @NOTE: testing
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )


    import sys
    log_file = open('/mnt/imgnet/job/job_log.txt', 'w')
    stdout = log_file
    stderr = log_file

    executor.update_parameters(name="deit")
    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir
    trainer = Trainer(args)
    #trainer()
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)





    # Define the log file paths
    log_file_out = f'/mnt/imgnet/job/{job.job_id}_0_log.out'
    log_file_err = f'/mnt/imgnet/job/{job.job_id}_0_log.err'


    #input("press q to kill the job")
    #job.cancel()
    job.wait()  # Wait for the job to finish
    log_file.close()


    with open(log_file_err, 'r') as err_file:
        print('======================================== ERROR FILE ========================================')
        print(err_file.read())
    with open(log_file_out, 'r') as out_file:
        print('======================================== OUTPUT FILE ========================================')
        print(out_file.read())




    #subprocess.run(["xfce4-terminal", "--hold", "--command=python "+ script_path])

if __name__ == "__main__":
    main()
