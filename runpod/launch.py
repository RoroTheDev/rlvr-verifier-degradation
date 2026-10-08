"""Launch, watch, fetch from, and stop RunPod pods that run runpod/verl_run.sh.

Standard library only. The API key is read from the RUNPOD_API_KEY environment
variable and is never written anywhere.

    export RUNPOD_API_KEY=...            # PowerShell: $env:RUNPOD_API_KEY = "..."

    python runpod/launch.py launch --run-name smoke --ref main \\
        --env MIXUP_MODE=persistent --env MIXUP_TPR=0.9 --env MIXUP_FPR=0.2 --env SEED=1

    python runpod/launch.py watch  --pod POD --run-name smoke
    python runpod/launch.py fetch  --pod POD --run-name smoke --out runs_local/smoke
    python runpod/launch.py stop   --pod POD

Pods are stateless: each launch rebuilds verl from pinned commits (~2-4 min) on the
container's local disk. There is deliberately no persistent network volume: on
Secure Cloud /workspace is a network FUSE filesystem, which made the env build
~3x slower, and a cached env saved little. (Startup hangs seen earlier were not the
filesystem -- they were verl's TransferQueue using up Ray's CPUs; verl_run.sh
shrinks it via TQ_STORAGE_UNITS.)

A pod that never starts (proxy answers an empty 404 for 5+ minutes, no public IP in
the API) is a bad host: terminate it and relaunch, optionally with --dc.

Always `fetch` before `stop`: a stopped pod's files are gone and the proxy goes
dark within seconds of termination.

Pin --ref to a tag, and pass --env VERL_REF=<sha>, for any run whose numbers will
be reported. Without VERL_REF the pod installs whatever verl `main` is that day,
and verl moves fast. The pod records both resulting shas in manifest.txt.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
# RunPod sits behind Cloudflare, which rejects urllib's default "Python-urllib/3.x"
# User-Agent with HTTP 403 / error 1010. Any explicit UA is accepted.
USER_AGENT = "rlvr-verifier-degradation-runpod-launcher/1.0"
DEFAULT_IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1290-torch291-ubuntu2404"


def _key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def _api(method: str, path: str, body=None):
    req = urllib.request.Request(
        API + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API {method} {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def _proxy(pod: str, rel: str) -> str:
    return f"https://{pod}-8080.proxy.runpod.net/{rel}"


def _get(url: str, timeout: int = 20):
    """Bytes, or None if the file does not exist / the pod is not serving yet."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return None


def cmd_launch(a):
    with open(os.path.join(HERE, "verl_run.sh"), encoding="utf-8", newline="") as f:
        script = f.read().replace("\r\n", "\n")

    env = {"REPO_REF": a.ref, "RUN_NAME": a.run_name}
    for pair in a.env:
        k, sep, v = pair.partition("=")
        if not sep:
            sys.exit(f"--env expects KEY=VALUE, got {pair!r}")
        env[k] = v

    # Every run logs to Weights & Biases when WANDB_API_KEY is available locally
    # (more runs, more data); --no-wandb opts out, --wandb makes a missing key an error.
    key = os.environ.get("WANDB_API_KEY")
    use_wandb = a.wandb if a.wandb is not None else bool(key)
    if use_wandb:
        # The key is read from the local environment only, never from the command
        # line, and launch.py never prints the pod's env. It does end up in the
        # pod's env on RunPod, so use a key you can revoke.
        if not key:
            sys.exit("--wandb needs WANDB_API_KEY set in your local shell")
        env["WANDB"] = "1"
        env["WANDB_API_KEY"] = key
        for k in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_TAGS"):
            if os.environ.get(k) and k not in env:
                env[k] = os.environ[k]

    payload = {
        "name": f"verl-{a.run_name}",
        "cloudType": a.cloud,
        "gpuTypeIds": [a.gpu],
        "gpuCount": 1,
        "imageName": a.image,
        "containerDiskInGb": a.container_disk,
        "ports": ["8080/http", "22/tcp"],
        "dockerEntrypoint": ["/bin/bash", "-c", script],
        "env": env,
        "interruptible": False,
    }
    if a.cuda_versions:
        # verl's torch wheels are built for CUDA 13 and need NVIDIA driver >= 580. Pods on
        # older hosts boot fine and nvidia-smi works, but torch then sees zero GPUs.
        payload["allowedCudaVersions"] = [v.strip() for v in a.cuda_versions.split(",") if v.strip()]
    if a.volume_id:
        if not a.dc:
            sys.exit("--volume-id requires --dc (the volume's data center)")
        payload["networkVolumeId"] = a.volume_id
        payload["dataCenterIds"] = [a.dc]
    else:
        payload["volumeInGb"] = a.volume_gb  # outputs only; the verl env lives on the container disk
        if a.dc:
            payload["dataCenterIds"] = [a.dc]

    pod = _api("POST", "/pods", payload)
    print(json.dumps({
        "pod": pod["id"],
        "costPerHr": pod.get("costPerHr"),
        "machine": pod.get("machine", {}).get("dataCenterId") or pod.get("machine", {}).get("location"),
        "gpu": pod.get("machine", {}).get("gpuTypeId"),
        "wandb": use_wandb,
        "progress": _proxy(pod["id"], f"runs/{a.run_name}/progress.txt"),
    }, indent=2))


def cmd_watch(a):
    seen = ""
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        data = _get(_proxy(a.pod, f"runs/{a.run_name}/progress.txt"))
        text = data.decode(errors="replace") if data else ""
        if text != seen:
            print(text[len(seen):] if text.startswith(seen) else text, end="", flush=True)
            seen = text
        if "STAGE:DONE" in text:
            return
        time.sleep(a.interval)
    sys.exit(f"timed out after {a.timeout}s; last progress:\n{seen or '(pod not serving yet)'}")


def cmd_fetch(a):
    os.makedirs(a.out, exist_ok=True)
    base = f"runs/{a.run_name}/"
    saved = []
    for name in ("progress.txt", "manifest.txt", "train_log.txt", "team_repo_clone.log", "data_prep.log", "verl_clone.log", "uv_install.log"):
        data = _get(_proxy(a.pod, base + name), timeout=60)
        if data is not None:
            with open(os.path.join(a.out, name), "wb") as f:
                f.write(data)
            saved.append(name)
    for sub in ("mixup_logs", "rollouts", "ray_logs"):
        listing = _get(_proxy(a.pod, f"{base}{sub}/"))
        if not listing:
            continue
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for fname in sorted(set(re.findall(r'href="([^"/?]+)"', listing.decode(errors="replace")))):
            data = _get(_proxy(a.pod, f"{base}{sub}/{fname}"), timeout=120)
            if data is not None:
                with open(os.path.join(a.out, sub, fname), "wb") as f:
                    f.write(data)
                saved.append(f"{sub}/{fname}")
    print(f"saved {len(saved)} files to {a.out}:")
    for s in saved:
        print("  " + s)
    if not saved:
        sys.exit("nothing fetched -- wrong pod/run name, or the pod is already gone")


def cmd_stop(a):
    _api("DELETE", f"/pods/{a.pod}")
    remaining = _api("GET", "/pods")
    print(f"terminated {a.pod}; pods still on the account: {[p['id'] for p in remaining]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("launch")
    l.add_argument("--run-name", required=True)
    l.add_argument("--ref", default="main", help="branch/tag/commit of this repo to run (default: main)")
    l.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    l.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    l.add_argument("--volume-id", help="persistent network volume holding the cached verl env")
    l.add_argument("--dc", help="data center to pin the pod to, e.g. EU-RO-1 (required with --volume-id)")
    l.add_argument("--volume-gb", type=int, default=30, help="size of /workspace (outputs only; the env lives on container disk)")
    l.add_argument("--container-disk", type=int, default=80, help="local disk holding verl, its venv and caches")
    l.add_argument("--image", default=DEFAULT_IMAGE)
    l.add_argument("--cuda-versions", default="13.0",
                   help="only schedule on hosts offering these CUDA versions (13.0 = driver >= 580, the API's highest value); '' disables the filter")
    l.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="forwarded to verl_run.sh / the reward function (repeatable)")
    l.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                   help="log to Weights & Biases (default: on whenever WANDB_API_KEY is set locally; "
                        "optional WANDB_ENTITY / WANDB_RUN_GROUP / WANDB_TAGS are forwarded too)")
    l.set_defaults(fn=cmd_launch)

    w = sub.add_parser("watch")
    w.add_argument("--pod", required=True)
    w.add_argument("--run-name", required=True)
    w.add_argument("--interval", type=int, default=20)
    w.add_argument("--timeout", type=int, default=3600)
    w.set_defaults(fn=cmd_watch)

    f = sub.add_parser("fetch")
    f.add_argument("--pod", required=True)
    f.add_argument("--run-name", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("stop")
    s.add_argument("--pod", required=True)
    s.set_defaults(fn=cmd_stop)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
