"""Launch, watch, fetch from, and stop RunPod pods that run runpod/verl_run.sh.

Standard library only. The API key is read from the RUNPOD_API_KEY environment
variable and is never written anywhere.

    export RUNPOD_API_KEY=...            # PowerShell: $env:RUNPOD_API_KEY = "..."

    # one-time: a persistent volume so the verl environment is built once
    #   (create it in the RunPod console or POST /networkvolumes; note its id and
    #   data center -- a pod can only attach a volume from its own data center)

    python runpod/launch.py launch --volume-id VOL --dc EU-RO-1 \\
        --run-name smoke --ref main \\
        --env MIXUP_MODE=persistent --env MIXUP_TPR=0.9 --env MIXUP_FPR=0.2 --env SEED=1

    python runpod/launch.py watch  --pod POD --run-name smoke
    python runpod/launch.py fetch  --pod POD --run-name smoke --out runs_local/smoke
    python runpod/launch.py stop   --pod POD

Always `fetch` before `stop`: a stopped pod's files are gone (only the volume
survives), and the proxy goes dark within seconds of termination.

Pin --ref to a tag for any run whose numbers will be reported; the pod records
the resulting commit sha and the verl sha in manifest.txt.
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
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
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
        with urllib.request.urlopen(url, timeout=timeout) as resp:
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
    if a.volume_id:
        if not a.dc:
            sys.exit("--volume-id requires --dc (the volume's data center)")
        payload["networkVolumeId"] = a.volume_id
        payload["dataCenterIds"] = [a.dc]
    else:
        payload["volumeInGb"] = a.volume_gb  # ephemeral: the env is rebuilt every launch (~8 min)

    pod = _api("POST", "/pods", payload)
    print(json.dumps({
        "pod": pod["id"],
        "costPerHr": pod.get("costPerHr"),
        "machine": pod.get("machine", {}).get("dataCenterId") or pod.get("machine", {}).get("location"),
        "gpu": pod.get("machine", {}).get("gpuTypeId"),
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
    for name in ("progress.txt", "manifest.txt", "train_log.txt", "team_repo_clone.log", "data_prep.log"):
        data = _get(_proxy(a.pod, base + name), timeout=60)
        if data is not None:
            with open(os.path.join(a.out, name), "wb") as f:
                f.write(data)
            saved.append(name)
    for sub in ("mixup_logs", "rollouts"):
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
    l.add_argument("--dc", help="data center of that volume, e.g. EU-RO-1")
    l.add_argument("--volume-gb", type=int, default=150, help="size of the ephemeral volume when no --volume-id")
    l.add_argument("--container-disk", type=int, default=20)
    l.add_argument("--image", default=DEFAULT_IMAGE)
    l.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="forwarded to verl_run.sh / the reward function (repeatable)")
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
