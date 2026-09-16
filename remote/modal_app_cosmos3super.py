"""Modal app for NVIDIA Cosmos 3 **Super** (the full model; ~132 GB of bf16
weights, `nvidia/Cosmos3-Super`, public, OpenMDW-1.1) -- backs
`vitals/adapters/cosmos3.py` with `app_name="vitals-cosmos3super"`. Its own
app, separate from `vitals-cosmos3` (Nano), because it cannot run the way
Nano does: Super does not fit one GPU, and the diffusers docs are explicit
(2026-09-12, read directly, quoted where it matters) that the ONLY
multi-GPU path is tensor parallelism through their own helper module:

  "`Super`'s ~120 GB of weights do not fit on one 96 GB GPU, so it needs
   TP" ... "Cosmos 3 does **not** wire CP into the transformer or the
   declarative `enable_parallelism()` path ... the implementation lives in
   `examples/cosmos3/cosmos_parallel.py`" ... "Load the pipeline
   configuration and components on CPU, apply TP *before* moving the
   pipeline to the rank-local GPU, switch to the `native` backend ... Do
   not use `device_map` for this flow." ... "for TP, the degree must
   divide the KV heads (8)" ... "Weights are loaded to CPU and sharded
   onto the GPUs layer by layer, so the full model is never materialized
   on a single device."

So `image2video` here launches `torchrun --nproc_per_node=4` on a small
worker (the docs' own modular-pipeline snippet, image-to-video variant)
inside a 4x H100-80GB container (TP=4: 4 divides 8 KV heads; ~33 GB of
weights per rank + activations at 480p). Rank 0 writes the frames to a
file the function returns. Everything else matches the Nano app: same
frozen prompt wrapped as `json.dumps({"scene": prompt})` (no LLM
upsampling -- AGENT.md §11), 24 fps, safety checker ON (mandatory under
the license; the guardrail's nltk data is staged off the symlinked
volume mount exactly as the Nano app had to), HF token for the gated
guardrail repo via the `vitals-cosmos` secret.

Unmeasured until the first real call (the same "first call is a
debugging pass" rule as every model here): latency, whether the modular
pipeline's image-to-video call takes `image=` exactly as the task
pipeline does, and whether TP=4 leaves enough headroom at 640x480 for 49
frames. Cost is ~4x a Nano call per minute; the CLI helper module is
pulled from diffusers `main` at image build time (unpinned, same
acceptance as the Nano app).

Usage:
    modal deploy remote/modal_app_cosmos3super.py
    python3 remote/call_cosmos3.py --app vitals-cosmos3super download_checkpoints
    python3 remote/call_cosmos3.py --app vitals-cosmos3super image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "H100:4"
TP_DEGREE = 4
MODEL_ID = "nvidia/Cosmos3-Super"

app = modal.App("vitals-cosmos3super")

cosmos3super_cache = modal.Volume.from_name("vitals-cosmos3super-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",
        "ftfy",
        "pillow",
        "numpy",
        "cosmos_guardrail",   # mandatory safety checker (see the Nano app's docstring)
    )
    .pip_install("git+https://github.com/huggingface/diffusers")
    # The TP/CP helpers are NOT part of the pip package -- they live in the
    # repo's examples/ directory (docs, quoted above). Cloned at the same
    # unpinned main as the package so the two cannot disagree by version.
    .run_commands("git clone --depth 1 https://github.com/huggingface/diffusers /root/diffusers")
    .env({"HF_HOME": "/cache/hf"})
)

# The torchrun worker: the diffusers docs' own modular-pipeline snippet, made
# concrete for image-to-video, reading its inputs from files so the Modal
# function never has to serialize a PIL image across a subprocess boundary.
WORKER_PY = r'''
import json, os, sys, numpy as np, torch, torch.distributed as dist
from PIL import Image
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from torch.distributed.device_mesh import init_device_mesh
sys.path.insert(0, "/root/diffusers/examples/cosmos3")
from cosmos_parallel import enable_cosmos3_tensor_parallel, enable_cosmos3_flash_attention

job = json.load(open(sys.argv[1]))
tp = int(job["tp_degree"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (tp, 1), mesh_dim_names=("tp", "cp"))

pipe = Cosmos3OmniModularPipeline.from_pretrained(job["model_id"])
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()                      # mandatory under the license -- never disabled
enable_cosmos3_tensor_parallel(pipe.transformer, mesh["tp"])   # shard weights -> GPUs, BEFORE .to()
pipe.to(f"cuda:{local_rank}")
pipe.transformer.set_attention_backend("native")
enable_cosmos3_flash_attention(pipe.transformer)  # GQA-safe dense attention (docs: TP without CP)
scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False)
pipe.update_components(scheduler=scheduler)

generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(int(job["seed"]))
videos = pipe(
    prompt=job["prompt"], image=Image.open(job["image_path"]).convert("RGB"),
    num_frames=int(job["num_frames"]), height=int(job["height"]), width=int(job["width"]), fps=24.0,
    num_inference_steps=int(job["num_inference_steps"]), guidance_scale=float(job["guidance_scale"]),
    generator=generator, output="videos",
)
if dist.get_rank() == 0:
    frames = np.asarray(videos)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.dtype != np.uint8:
        frames = (np.asarray(frames, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    np.save(job["out_path"], frames)
dist.barrier()
dist.destroy_process_group()
'''


def _stage_guardrail_nltk_data(hf_home="/cache/hf", dst="/root/nltk_data"):
    """Same repair the Nano app needed (its docstring has the incident):
    the Modal mount point is a symlink and nltk's hardened opener refuses
    it, so the guardrail's nltk_data is copied to a real local dir and
    NLTK_DATA exported before any nltk import."""
    import glob, os, shutil
    srcs = glob.glob(f"{hf_home}/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots/*/blocklist/nltk_data")
    if not srcs:
        raise RuntimeError("guardrail nltk_data not found in the HF cache -- run download_checkpoints first")
    shutil.copytree(srcs[0], dst, symlinks=False, dirs_exist_ok=True)
    os.environ["NLTK_DATA"] = dst
    return dst


@app.function(image=image, gpu="H100", volumes={"/cache": cosmos3super_cache}, timeout=7200,
              secrets=[modal.Secret.from_name("vitals-cosmos")])
def download_checkpoints():
    """One-time: pulls the ~132 GB Super checkpoint plus the (gated)
    guardrail models into the persistent volume. CPU-side load only (a
    download, not an inference); a single GPU is attached only so the
    container class matches the inference tier's disk/network profile."""
    import torch
    from diffusers import Cosmos3OmniModularPipeline
    pipe = Cosmos3OmniModularPipeline.from_pretrained(MODEL_ID)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.enable_safety_checker()
    cosmos3super_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cosmos3super_cache}, timeout=3600,
              secrets=[modal.Secret.from_name("vitals-cosmos")])
def image2video(image, prompt: str, seed: int = 0, num_frames: int = 49,
                height: int = 480, width: int = 640,
                num_inference_steps: int = 35, guidance_scale: float = 6.0):
    """Runs ONE Cosmos 3 Super image-to-video generation under TP=4 and
    returns frames as a raw (T,H,W,3) uint8 numpy array -- the same
    contract as the Nano app's `image2video`, so `vitals/adapters/
    cosmos3.py` serves both apps unchanged."""
    import json, os, subprocess, tempfile
    import numpy as np
    from PIL import Image

    print("guardrail nltk_data staged at", _stage_guardrail_nltk_data())
    work = tempfile.mkdtemp()
    image_path = os.path.join(work, "input.png"); Image.fromarray(np.asarray(image)).save(image_path)
    out_path = os.path.join(work, "frames.npy")
    worker = os.path.join(work, "worker.py"); open(worker, "w").write(WORKER_PY)
    job = dict(model_id=MODEL_ID, tp_degree=TP_DEGREE, prompt=json.dumps({"scene": prompt}),
               image_path=image_path, out_path=out_path, seed=seed, num_frames=num_frames,
               height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    job_path = os.path.join(work, "job.json"); json.dump(job, open(job_path, "w"))

    env = dict(os.environ, NLTK_DATA=os.environ["NLTK_DATA"], PYTHONUNBUFFERED="1")
    cmd = ["torchrun", f"--nproc_per_node={TP_DEGREE}", worker, job_path]
    print("launching:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = lambda s: "\n".join(s.strip().splitlines()[-40:])
    print("torchrun stdout (tail):\n" + tail(proc.stdout))
    if proc.returncode != 0:
        raise RuntimeError(f"torchrun exited {proc.returncode}\nstderr (tail):\n{tail(proc.stderr)}")
    frames = np.load(out_path)
    cosmos3super_cache.commit()
    return frames
