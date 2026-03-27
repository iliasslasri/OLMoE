#!/usr/bin/env python3
"""launch_eval2.py — Phase-2 eval launcher via SSH/paramiko.

Steps
-----
  1. Generate phase-2 configs locally (runs generate_eval_configs2.py)
  2. Upload configs + job script + train script to remote server
  3. Submit one sbatch job per config

One sbatch job per config keeps every run within the 36h wall-time limit:
  Group D  ~5 h/run   (5 runs)
  Group E  ~14 h/run  (6 runs)
  Group F  ~28 h/run  (4 runs)

Usage
-----
  # Dry-run (default) — shows what would be submitted:
  python launch_eval2.py

  # Actually submit all groups:
  python launch_eval2.py --submit

  # Submit only group D:
  python launch_eval2.py --group D --submit
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

try:
    import paramiko
except ImportError:
    sys.exit("paramiko not found. Install it with:  pip install paramiko")

# ── Remote server ──────────────────────────────────────────────────────────────
REMOTE_HOST  = "137.194.132.200"
REMOTE_USER  = "zakil-22"
REMOTE_PORT  = 22
REMOTE_OLMOE = "/home/infres/zakil-22/OLMoE"

# ── Local paths ────────────────────────────────────────────────────────────────
LOCAL_ROOT   = pathlib.Path(__file__).parent.resolve()
CONFIGS_DIR  = LOCAL_ROOT / "configs" / "eval2"
JOB_SCRIPT   = LOCAL_ROOT / "job_eval2_single.sh"
TRAIN_SCRIPT = LOCAL_ROOT / "train_server_grid.py"
GEN_SCRIPT   = LOCAL_ROOT / "generate_eval_configs2.py"


# ── SSH helpers ────────────────────────────────────────────────────────────────

def make_ssh_client() -> paramiko.SSHClient:
    """Connect using SSH agent, key files, or REMOTE_PASSWORD env var."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 1. Try SSH agent + key files
    try:
        client.connect(
            REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER,
            timeout=30, allow_agent=True, look_for_keys=True,
        )
        print(f"Connected to {REMOTE_USER}@{REMOTE_HOST} (key auth)")
        return client
    except paramiko.AuthenticationException:
        pass

    # 2. Explicit key paths
    for key_path in [
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ecdsa"),
    ]:
        if os.path.exists(key_path):
            try:
                client.connect(
                    REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER,
                    key_filename=key_path, timeout=30,
                )
                print(f"Connected via {key_path}")
                return client
            except (paramiko.AuthenticationException, Exception):
                continue

    # 3. Password fallback (read from env var to avoid hardcoding)
    password = os.environ.get("REMOTE_PASSWORD")
    if password:
        client.connect(
            REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER,
            password=password, timeout=30,
            allow_agent=False, look_for_keys=False,
        )
        print(f"Connected to {REMOTE_USER}@{REMOTE_HOST} (password auth)")
        return client

    raise RuntimeError(
        f"Authentication failed for {REMOTE_USER}@{REMOTE_HOST}.\n"
        "Options:\n"
        "  1. Set up key auth:   ssh-copy-id zakil-22@137.194.132.200\n"
        "  2. Pass password:     REMOTE_PASSWORD=xxx python3 launch_eval2.py --submit"
    )


def run_remote(client: paramiko.SSHClient, cmd: str, check: bool = True):
    """Run a shell command on the remote host and return (stdout, stderr, rc)."""
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    rc  = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(
            f"Remote command failed (exit {rc}):\n  cmd : {cmd}\n  err : {err}"
        )
    return out, err, rc


def upload_file(sftp: paramiko.SFTPClient, local: pathlib.Path, remote: str):
    """Upload a single file, creating the remote parent directory if needed."""
    remote_dir = str(pathlib.PurePosixPath(remote).parent)
    # mkdir -p equivalent via SFTP
    parts = pathlib.PurePosixPath(remote_dir).parts
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        try:
            sftp.mkdir(current)
        except OSError:
            pass  # already exists
    sftp.put(str(local), remote)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 eval launcher via paramiko/SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--group", choices=["D", "E", "F", "all"], default="all",
        help="Which group(s) to launch (default: all)",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit sbatch jobs (default: dry-run)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Submit at most N jobs (useful to test one before the rest)",
    )
    args = parser.parse_args()

    groups = ["D", "E", "F"] if args.group == "all" else [args.group]

    # ── Step 1: Generate configs locally ──────────────────────────────────────
    print("=" * 64)
    print("Step 1 — Generating phase-2 eval configs locally")
    print("=" * 64)

    if not GEN_SCRIPT.exists():
        sys.exit(f"ERROR: {GEN_SCRIPT} not found")

    result = subprocess.run(
        [sys.executable, str(GEN_SCRIPT)], cwd=LOCAL_ROOT,
    )
    if result.returncode != 0:
        sys.exit(f"Config generation failed (exit {result.returncode})")

    # Collect configs for selected groups
    all_configs: list[pathlib.Path] = []
    for group in groups:
        group_dir = CONFIGS_DIR / group
        configs = sorted(group_dir.glob("*.yml"))
        if not configs:
            sys.exit(f"ERROR: No configs found in {group_dir}")
        all_configs.extend(configs)
        print(f"  Group {group}: {len(configs)} config(s)")

    print(f"\nTotal: {len(all_configs)} config(s) across group(s) {', '.join(groups)}")
    if args.limit:
        all_configs = all_configs[: args.limit]
        print(f"  (limited to {args.limit} by --limit)")

    # ── Dry-run ────────────────────────────────────────────────────────────────
    if not args.submit:
        print("\n[DRY RUN]  Would submit:")
        for cfg in all_configs:
            remote_cfg = f"{REMOTE_OLMOE}/configs/eval2/{cfg.parent.name}/{cfg.name}"
            print(f"  sbatch job_eval2_single.sh  {remote_cfg}")
        print(
            "\nRe-run with --submit to actually upload and submit.\n"
            "  python launch_eval2.py --submit\n"
            "  python launch_eval2.py --group D --submit"
        )
        return

    # ── Step 2: Connect and upload ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Step 2 — Uploading files to remote server")
    print("=" * 64)

    client = make_ssh_client()
    sftp   = client.open_sftp()

    # Create remote directory structure
    for group in groups:
        run_remote(client, f"mkdir -p {REMOTE_OLMOE}/configs/eval2/{group}")
    run_remote(client, f"mkdir -p {REMOTE_OLMOE}/logs/eval2")
    print("Remote directories OK.")

    # Upload job script
    remote_job = f"{REMOTE_OLMOE}/job_eval2_single.sh"
    print(f"Uploading {JOB_SCRIPT.name} ...")
    upload_file(sftp, JOB_SCRIPT, remote_job)
    run_remote(client, f"chmod +x {remote_job}")

    # Upload train runner (reuse train_server_grid.py — same interface)
    remote_train = f"{REMOTE_OLMOE}/train_server_grid.py"
    print(f"Uploading {TRAIN_SCRIPT.name} ...")
    upload_file(sftp, TRAIN_SCRIPT, remote_train)

    # Upload YAML configs
    print(f"Uploading {len(all_configs)} config file(s)...")
    for cfg in all_configs:
        remote_cfg = f"{REMOTE_OLMOE}/configs/eval2/{cfg.parent.name}/{cfg.name}"
        sftp.put(str(cfg), remote_cfg)
    print("Upload done.")

    # ── Step 3: Submit sbatch jobs ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Step 3 — Submitting sbatch jobs")
    print("=" * 64)

    submitted: list[tuple[str, str, str]] = []   # (group, stem, job_id)
    for cfg in all_configs:
        remote_cfg = f"{REMOTE_OLMOE}/configs/eval2/{cfg.parent.name}/{cfg.name}"
        cmd = f"cd {REMOTE_OLMOE} && sbatch {remote_job} {remote_cfg}"
        out, _, _ = run_remote(client, cmd)
        # sbatch prints "Submitted batch job <id>"
        job_id = out.split()[-1] if out else "?"
        group  = cfg.parent.name
        print(f"  [{group}] {cfg.stem:<46}  job_id={job_id}")
        submitted.append((group, cfg.stem, job_id))
        time.sleep(0.3)   # brief pause between submissions

    sftp.close()
    client.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"Submitted {len(submitted)} job(s):")
    for group, name, jid in submitted:
        print(f"  [{group}] {name}  →  job {jid}")

    print("\nUseful commands on the server:")
    print(f"  squeue -u {REMOTE_USER}                           # check queue")
    print(f"  ls {REMOTE_OLMOE}/logs/eval2/                     # job logs")
    print(f"  tail -f {REMOTE_OLMOE}/logs/eval2/eval2_<JOB>.out # live log")
    print("=" * 64)


if __name__ == "__main__":
    main()
