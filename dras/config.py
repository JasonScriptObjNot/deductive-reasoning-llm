from dataclasses import dataclass, field


@dataclass
class Config:
    # model
    model_id: str = "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
    max_seq_length: int = 1024

    # lora — matches existing setup exactly
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # training
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    bf16: bool = True
    packing: bool = False

    # retriever
    retriever_backend: str = "dense"   # "dense" | "bm25"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # path to fine-tuned retriever; empty string = use embed_model from HuggingFace
    retriever_model_path: str = ""
    retrieval_k: int = 5

    # inference
    repetition_penalty: float = 1.3
    max_new_tokens: int = 128

    # loop termination
    max_iterations: int = 20
    goal_sim_threshold: float = 0.85   # lowered from 0.90 — fixes B5-style paraphrase misses
    dedup_threshold: float = 0.95

    # validation
    validate_steps: bool = True        # run validator before adding each generated step to store
    validate_manual: bool = True       # run validator on user-supplied seed premises

    # lookahead: when a valid greedy step's 1-step future goal_sim is below lookahead_threshold,
    # generate lookahead_k sampled alternatives and keep whichever scores highest.
    # 0 disables lookahead entirely (default = off, opt-in).
    lookahead_k: int = 0
    lookahead_threshold: float = 0.55  # trigger lookahead when greedy's future sim is below this

    # backward chaining: when stuck for backchain_k consecutive rejections, identify
    # the nearest applicable bridge rule and re-derive the missing step via focused
    # two-premise inference (anchor_fact + bridge_rule only, no distractors).
    # This mirrors how mathematicians target intermediate lemmas when the final goal
    # is unreachable in one step. 0 disables. Default 6 fires after the sampling
    # fallback has had one full cycle (3 greedy + 3 sampled rejects).
    backchain_k: int = 6

    # exhaustive pairwise search: after pair_search_k consecutive rejections (all
    # other recovery mechanisms exhausted), try every ordered pair of premises in
    # the store via focused 2-premise inference, BFS-expanding valid new steps.
    # Expensive (O(N²) inference calls) but guaranteed to find any 1-step reachable
    # fact from the current store. 0 disables. Recommended: 2 × backchain_k.
    pair_search_k: int = 0

    # goal projection: backward sub-goal synthesis from the final goal.
    # When stuck, finds goal-proximate rules and asks the model to project the
    # antecedent of each as an intermediate sub-goal via infer(model, goal, [rule]).
    # Temporarily switches the retrieval anchor from the far-away final goal to the
    # projected sub-goal, giving the forward chain a closer waypoint to aim for.
    # Fires at this many consecutive rejections (0 disables).
    # Recommended: backchain_k + 3 (= 9 when backchain_k=6) — between the first
    # and second backchain trigger, before pairwise search kicks in.
    goal_proj_k: int = 0

    # paths
    data_dir: str = "data"
    output_dir: str = "outputs"
