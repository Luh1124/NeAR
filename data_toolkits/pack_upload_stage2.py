#!/usr/bin/env python3
"""
Pack NeAR Stage-2 eevee renders from TOS into tar shards and upload to ModelScope.

Pipeline (producer + N parallel consumers):
  Producer thread:   download batch → clean excluded → pack tar → enqueue
  Consumer threads:  dequeue → upload → delete local files → save state
"""

import csv
import io
import json
import os
import queue
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tos

# ── Config ────────────────────────────────────────────────────────────────────

TOS_BUCKET        = "lhtest"
TOS_KEY_BASE      = "3diclight/3diclight_even_8w9/renders_3diclight_neural_graffer_0309/eevee/eevee"
TOS_AK            = os.environ["TOS_AK"]
TOS_SK            = os.environ["TOS_SK"]
TOS_ENDPOINT      = "tos-cn-beijing.volces.com"
TOS_REGION        = "cn-beijing"

BUFFER_DIR        = Path("/tmp/near_buffer")
SHARD_DIR         = Path("/tmp/near_shards")
STATE_FILE        = Path(__file__).parent / "pack_state.json"
CSV_FILE          = Path("/root/code/zeroverse/metadatas/93528_eevee_and_cycles_filtered_2000_thin001_eevee_to_88259.csv")

MODELSCOPE_REPO   = "luh0502/NeAR-dataset"
ACCESS_TOKEN      = os.environ["MODELSCOPE_ACCESS_TOKEN"]

OBJECTS_PER_SHARD = 10    # ~10.7GB per shard, 8826 shards total → fits relight/ ≤ 10000
DOWNLOAD_WORKERS  = 3     # parallel object downloads within one shard
FILE_DL_WORKERS   = 50    # parallel file downloads within one object (SDK connection pool)
UPLOAD_WORKERS    = 2     # parallel ModelScope upload threads
QUEUE_SIZE        = 2     # shards buffered between producer and consumers

EXCLUDE_DIRS = {"ao", "glossycol", "glossydir", "diffdir", "albedo", "env"}

# ── TOS client (shared, thread-safe) ─────────────────────────────────────────

_tos_client: tos.TosClientV2 = None

def get_tos_client() -> tos.TosClientV2:
    global _tos_client
    if _tos_client is None:
        _tos_client = tos.TosClientV2(
            ak=TOS_AK, sk=TOS_SK,
            endpoint=TOS_ENDPOINT, region=TOS_REGION,
            max_connections=500,
            socket_timeout=300,
            high_latency_log_threshold=0,
        )
    return _tos_client

# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        # migrate: convert completed_shards list back to set
        s["completed_shards"] = set(s.get("completed_shards", []))
        return s
    return {"next_idx": 0, "next_shard": 0, "skipped": [],
            "uploaded": 0, "completed_shards": set()}


def save_state(state: dict):
    """Must be called while holding state_lock."""
    out = dict(state)
    out["completed_shards"] = sorted(state["completed_shards"])
    STATE_FILE.write_text(json.dumps(out, indent=2))

# ── SHA256 list ───────────────────────────────────────────────────────────────

def load_sha256_list() -> list:
    with open(CSV_FILE) as f:
        return [row["sha256"].strip() for row in csv.DictReader(f)]

# ── Download ──────────────────────────────────────────────────────────────────

def download_object(sha256: str) -> bool:
    dest = BUFFER_DIR / sha256
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{TOS_KEY_BASE}/{sha256}/"
    client = get_tos_client()
    t0 = time.time()
    try:
        # list all keys under this object's prefix
        keys: list = []
        token: str = ""
        while True:
            out = client.list_objects_type2(
                bucket=TOS_BUCKET, prefix=prefix,
                continuation_token=token or None, max_keys=1000,
            )
            keys.extend(
                obj.key for obj in out.contents
                if not obj.key.endswith("/")
                and obj.key[len(prefix):].split("/")[0] not in EXCLUDE_DIRS
            )
            if not out.is_truncated:
                break
            token = out.next_continuation_token or ""

        if not keys:
            print(f"    [SKIP] {sha256[:16]}...: not found on TOS")
            shutil.rmtree(dest, ignore_errors=True)
            return False

        def dl_one(key: str):
            rel = key[len(prefix):]          # strip object prefix
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            client.get_object_to_file(bucket=TOS_BUCKET, key=key, file_path=str(local))

        with ThreadPoolExecutor(max_workers=FILE_DL_WORKERS) as pool:
            list(pool.map(dl_one, keys))

        elapsed = time.time() - t0
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024 ** 2
        print(f"    [dl] {sha256[:16]}...  {size_mb:.0f} MB / {len(keys)} files  "
              f"{elapsed:.1f}s  ({size_mb/elapsed:.0f} MB/s)")
        return True

    except Exception as exc:
        print(f"    [SKIP] {sha256[:16]}...: {exc}")
        shutil.rmtree(dest, ignore_errors=True)
        return False


def fix_nested(dest: Path, sha256: str):
    """tosutil sometimes creates dest/sha256/sha256/ instead of dest/sha256/."""
    nested = dest / sha256
    if nested.is_dir():
        for item in nested.iterdir():
            item.rename(dest / item.name)
        nested.rmdir()


def clean_excluded(obj_dir: Path):
    for name in EXCLUDE_DIRS:
        d = obj_dir / name
        if d.exists():
            shutil.rmtree(d)

# ── Pack ──────────────────────────────────────────────────────────────────────

def pack_shard(sha256_list: list, shard_id: int) -> Path:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = SHARD_DIR / f"{shard_id:06d}.tar"
    manifest_bytes = json.dumps({"shard_id": shard_id, "objects": sha256_list}, indent=2).encode()

    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))
        for sha256 in sha256_list:
            obj_dir = BUFFER_DIR / sha256
            if obj_dir.exists():
                tf.add(obj_dir, arcname=sha256)

    size_gb = tar_path.stat().st_size / 1024 ** 3
    print(f"  [producer] Packed  {tar_path.name}: {size_gb:.2f} GB ({len(sha256_list)} objects)")
    return tar_path

# ── Upload ────────────────────────────────────────────────────────────────────

def upload_shard(api, tar_path: Path, shard_id: int, worker_id: int):
    repo_path = f"relight/{shard_id:06d}.tar"
    size_gb = tar_path.stat().st_size / 1024 ** 3
    print(f"  [upload-{worker_id}] Uploading {tar_path.name} ({size_gb:.2f} GB) -> {repo_path}")
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=repo_path,
        repo_id=MODELSCOPE_REPO,
        repo_type="dataset",
        commit_message=f"shard {shard_id:06d}",
    )
    elapsed = time.time() - t0
    print(f"  [upload-{worker_id}] Done {tar_path.name} in {elapsed/60:.1f} min "
          f"({size_gb*1024/elapsed:.0f} MB/s)")

# ── Producer ──────────────────────────────────────────────────────────────────

def producer(sha256_list, start_idx, start_shard, objects_per_shard, skipped_list, q):
    idx      = start_idx
    shard_id = start_shard

    while idx < len(sha256_list):
        batch = sha256_list[idx: idx + objects_per_shard]
        print(f"\n── Shard {shard_id:06d} │ objects {idx}–{idx+len(batch)-1} "
              f"│ {idx/len(sha256_list)*100:.1f}% ──")

        print(f"  Downloading {len(batch)} objects ({DOWNLOAD_WORKERS} parallel)...")
        downloaded = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(download_object, s): s for s in batch}
            for fut in as_completed(futures):
                sha256 = futures[fut]
                if fut.result():
                    fix_nested(BUFFER_DIR / sha256, sha256)
                    clean_excluded(BUFFER_DIR / sha256)
                    downloaded.append(sha256)
                    print(f"    [ok] {sha256[:16]}... ({len(downloaded)}/{len(batch)})")
                else:
                    skipped_list.append(sha256)

        if not downloaded:
            print("  All objects missing on TOS, skipping shard")
            idx += len(batch)
            q.put({"skip": True, "idx": idx, "shard_id": shard_id,
                   "skipped": list(skipped_list)})
            shard_id += 1
            continue

        tar_path = pack_shard(downloaded, shard_id)
        q.put({
            "skip": False,
            "tar_path": tar_path,
            "shard_id": shard_id,
            "downloaded": downloaded,
            "idx": idx + len(batch),
            "skipped": list(skipped_list),
        })
        idx      += len(batch)
        shard_id += 1

    # one sentinel per consumer thread
    for _ in range(UPLOAD_WORKERS):
        q.put(None)

# ── Consumer ──────────────────────────────────────────────────────────────────

def consumer_worker(worker_id, api, state, state_lock, q):
    while True:
        item = q.get()
        if item is None:
            break
        if item["skip"]:
            with state_lock:
                state["skipped"] = item["skipped"]
                save_state(state)
            continue

        tar_path   = item["tar_path"]
        shard_id   = item["shard_id"]
        downloaded = item["downloaded"]

        # retry once on commit conflict or transient error
        for attempt in range(3):
            try:
                upload_shard(api, tar_path, shard_id, worker_id)
                break
            except Exception as exc:
                wait = 30 * (attempt + 1)
                if attempt < 2:
                    print(f"  [upload-{worker_id}] Error: {exc} — retry in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        for sha256 in downloaded:
            shutil.rmtree(BUFFER_DIR / sha256, ignore_errors=True)
        tar_path.unlink(missing_ok=True)

        with state_lock:
            state["completed_shards"].add(shard_id)
            state["uploaded"] = len(state["completed_shards"])
            state["skipped"]  = item["skipped"]
            # next_idx / next_shard only used for resume — keep the high-water mark
            if item["idx"] > state["next_idx"]:
                state["next_idx"]   = item["idx"]
                state["next_shard"] = shard_id + 1
            save_state(state)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run one shard (1 object) then exit")
    args = parser.parse_args()

    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    sha256_list = load_sha256_list()
    print(f"Total objects in CSV: {len(sha256_list)}")

    from modelscope.hub.api import HubApi

    # one API instance per upload thread to avoid shared state issues
    apis = []
    for _ in range(UPLOAD_WORKERS):
        api = HubApi()
        api.login(ACCESS_TOKEN)
        apis.append(api)

    state = load_state()
    print(f"Resuming from idx={state['next_idx']}, shard={state['next_shard']}, "
          f"uploaded={state['uploaded']}, skipped={len(state['skipped'])}")

    objects_per_shard = 1 if args.test else OBJECTS_PER_SHARD
    skipped_list = list(state["skipped"])
    target_list  = sha256_list[:state["next_idx"] + objects_per_shard] if args.test else sha256_list

    q          = queue.Queue(maxsize=QUEUE_SIZE)
    state_lock = threading.Lock()

    prod = threading.Thread(
        target=producer,
        args=(target_list, state["next_idx"], state["next_shard"],
              objects_per_shard, skipped_list, q),
        daemon=True,
    )
    prod.start()

    consumers = [
        threading.Thread(
            target=consumer_worker,
            args=(i, apis[i], state, state_lock, q),
            daemon=True,
        )
        for i in range(UPLOAD_WORKERS)
    ]
    for c in consumers:
        c.start()
    for c in consumers:
        c.join()
    prod.join()

    print(f"\nDone. {state['uploaded']} shards uploaded, {len(state['skipped'])} objects skipped.")


if __name__ == "__main__":
    main()
