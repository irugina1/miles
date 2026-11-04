"""True on-policy GRPO for Qwen3-30B-A3B on one 8-GPU node (FSDP trainer, SGLang rollout).

The trainer is bitwise equal to the sampler; ``train/train_rollout_logprob_abs_diff`` is 0. See
examples/infra_features/true_on_policy/README.md.

    python examples/infra_features/true_on_policy/run_qwen3_30b_a3b.py --r3 \
        --model-dir /root/models --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl

``--mode debug_minimal`` runs a few short steps. ``--r3`` replays the sampler's expert routing.
``--async-rollout`` (with ``--no-colocate``) generates rollout i+1 while step i trains.
``--rollout-fp8-checkpoint`` and ``--moe-quant`` quantize the routed experts.
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.kernels.moe.mx_fused_moe import MOE_QUANT_ENV
from miles.true_on_policy import apply_true_on_policy_script_defaults, build_true_on_policy_launch_plan
from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    model_name: str = "Qwen3-30B-A3B-Base"
    model_dir: str = "/root/models"
    prompt_data: str = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
    input_key: str = "prompt"
    label_key: str = "label"
    num_gpus_per_node: int = 8
    # colocate: trainer and rollout share all GPUs; otherwise the trainer gets actor_num_gpus
    # and the remaining GPUs run one single-GPU SGLang engine each
    colocate: bool = True
    actor_num_gpus: int = 4
    max_tokens_per_gpu: int = 16384
    true_on_policy: bool = True
    true_on_policy_contract: str | None = None
    # experts-only block-FP8 copy of the model, served by the rollout engines
    rollout_fp8_checkpoint: str | None = None
    # W+A microscaling quantization of the routed experts on both sides
    moe_quant: Literal["mxfp8", "mxfp4", "nvfp4"] | None = None
    # rollout routing replay: training replays the experts the sampler selected
    r3: bool = False
    # one step of constant staleness: rollout i+1 is generated while step i trains (train_async.py)
    async_rollout: bool = False
    extra_args: str = ""

    def __post_init__(self):
        # Fields read by the true-on-policy launch plan.
        self.train_backend = "fsdp"
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.rollout_num_gpus_per_engine = 1
        apply_true_on_policy_script_defaults(self)


def execute(args: ScriptArgs) -> None:
    U = args.create_backend()
    debug = args.mode == "debug_minimal"
    model = f"{args.model_dir}/{args.model_name}"

    rollout_args = (
        f"--prompt-data {args.prompt_data} --input-key {args.input_key} --label-key {args.label_key} "
        "--apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {3 if debug else 3000} "
        f"--rollout-batch-size {8 if debug else 32} --n-samples-per-prompt {4 if debug else 8} "
        f"--global-batch-size {32 if debug else 256} "
        f"--rollout-max-response-len {512 if debug else 8192} --rollout-temperature 1 "
    )
    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    fsdp_args = (
        "--train-backend fsdp --no-enable-sample-ownership-checker --gradient-checkpointing "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )
    # rollout GPUs across all nodes
    if args.colocate:
        rollout_num_gpus = args.num_gpus_per_node * args.num_nodes
        placement_args = f"--colocate --actor-num-gpus-per-node {args.num_gpus_per_node} "
        sglang_mem = 0.6
    else:
        rollout_num_gpus = (args.num_gpus_per_node - args.actor_num_gpus) * args.num_nodes
        placement_args = f"--actor-num-gpus-per-node {args.actor_num_gpus} --rollout-num-gpus {rollout_num_gpus} "
        sglang_mem = 0.8
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {sglang_mem} --sglang-chunked-prefill-size 4096 "
    )
    if args.rollout_fp8_checkpoint is not None:
        # The engines serve the FP8 copy and still receive every weight sync; the FSDP trainer
        # reads its quantization_config to quantize synced weights and its own MoE forward.
        sglang_config = f"""
sglang:
  - name: default
    model_path: {args.rollout_fp8_checkpoint}
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: {rollout_num_gpus}
"""
        sglang_args += f"--sglang-config {command_utils.encode_pseudo_file(sglang_config)} "
    misc_args = f"--actor-num-nodes {args.num_nodes} --num-gpus-per-node {args.num_gpus_per_node} "
    if args.r3:
        misc_args += "--use-rollout-routing-replay "

    plan = build_true_on_policy_launch_plan(args)

    train_args = (
        f"--hf-checkpoint {model} "
        f"{rollout_args}{grpo_args}{optimizer_args}{fsdp_args}{placement_args}{sglang_args}{misc_args}"
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{plan.train_args}{args.extra_args} "
    )

    env_vars = dict(plan.env_vars)
    if args.moe_quant is not None:
        if args.rollout_fp8_checkpoint is not None:
            raise ValueError("--moe-quant and --rollout-fp8-checkpoint are alternatives")
        if not args.true_on_policy:
            raise ValueError("--moe-quant requires true on-policy")
        # read by the FSDP trainer and by the SGLang engines (Miles' SGLang plugin)
        env_vars[MOE_QUANT_ENV] = args.moe_quant

    if args.async_rollout and args.colocate:
        raise ValueError("--async-rollout needs --no-colocate")
    U.execute_train(
        train_args=train_args,
        train_script="train_async.py" if args.async_rollout else "train.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars=env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
