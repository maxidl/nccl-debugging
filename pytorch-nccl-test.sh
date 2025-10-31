#!/bin/bash
#SBATCH --job-name=pytorch-nccl-test
#SBATCH --partition=
#SBATCH --account=
#SBATCH --qos=
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time 0:05:00
#SBATCH --output=%x-%j.out
##SBATCH --exclude=

# SLURM PyTorch NCCL Multi-Node Test Script:
# An example SLURM batch script that tests PyTorch's NCCL functionality across multiple GPU nodes and 
# performs a bandwidth test.
# The script sets up a distributed PyTorch environment using torchrun and runs a test that verifies 
# NCCL initialization, inter-process communication barriers, bandwidth, and proper cleanup.
# Includes diagnostic output for troubleshooting multi-node GPU communication issues in HPC environments.

# Print job information
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_ACCOUNT: ${SLURM_JOB_ACCOUNT}"
echo "============================="

set -e

GPUS_PER_NODE=4

# NCCL flags to play with these.
export NCCL_DEBUG=WARN # INFO TRACE
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NVLS_ENABLE=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export GLOO_SOCKET_IFNAME=ib0
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_NET_GDR_READ=0
# export NCCL_P2P_DISABLE=1
# export TORCH_NCCL_BLOCKING_WAIT=0
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
# export TORCH_COMPILE_DISABLE=1
# export ACCELERATE_USE_CUDA_GRAPHS=0
# export TORCH_CUDAGRAPHS=0
# export TORCHDYNAMO_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_NET=IB
# export OMP_NUM_THREADS=2
# export NCCL_SHM_DISABLE=1
# # export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# export CUDA_LAUNCH_BLOCKING=1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((20000 + $SLURM_JOB_ID % 10000))

# Print job information
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_ACCOUNT: ${SLURM_JOB_ACCOUNT}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_NODEID: ${SLURM_NODEID}"
echo "============================="

LAUNCHER=(
    torchrun
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$SLURM_JOB_NUM_NODES"
    --rdzv_backend c10d
    --rdzv_id "$SLURM_JOB_ID"
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
)

export SCRIPT=pytorch-nccl-test.py

cat << EOT > $SCRIPT
import torch.distributed as dist
import torch
import socket
import os
import fcntl
import time
import math
import glob

SUCCESS = 0

def printflock(*msgs):
    """Process-safe printing using file lock on this script path."""
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs, flush=True)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

# -------------------- small utils --------------------
def percentile(values, q):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * (q / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)

def gather_all(obj):
    """Gather arbitrary Python objects from all ranks."""
    world_size = dist.get_world_size()
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out

def get_nodes_and_mapping():
    """Return (sorted_nodes, ranks_by_node (dict: node->list[ranks]), gpus_per_node_map)."""
    host = socket.gethostname()
    all_hosts = gather_all(host)
    nodes = sorted(set(all_hosts))
    by_node = {n: [] for n in nodes}
    for r, h in enumerate(all_hosts):
        by_node[h].append(r)
    gpus_per_node_map = {n: len(rs) for n, rs in by_node.items()}
    return nodes, by_node, gpus_per_node_map

# ----------- InfiniBand counters for wire throughput -----------
def _read_int(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def _ib_ports():
    """List ACTIVE IB ports as tuples (dev, port, rate_str)."""
    ports = []
    for dev in sorted(glob.glob("/sys/class/infiniband/mlx5_*")):
        pd = os.path.join(dev, "ports")
        if not os.path.isdir(pd):
            continue
        for p in sorted(os.listdir(pd)):
            try:
                with open(os.path.join(pd, p, "state")) as f:
                    state = f.read()
            except Exception:
                state = ""
            if "ACTIVE" not in state.upper():
                continue
            try:
                with open(os.path.join(pd, p, "rate")) as f:
                    rate = f.read().strip()  # e.g., "100 Gb/sec (4X HDR)"
            except Exception:
                rate = ""
            ports.append((dev.split("/")[-1], p, rate))
    return ports

def _ib_snapshot():
    """Take a snapshot of xmit/recv byte counters for ACTIVE ports."""
    snap = []
    for dev, port, rate in _ib_ports():
        cdir = f"/sys/class/infiniband/{dev}/ports/{port}/counters"
        # Prefer 64-bit *_ext if available; else legacy 32-bit words (4B each)
        tx = _read_int(os.path.join(cdir, "port_xmit_data_ext"))
        rx = _read_int(os.path.join(cdir, "port_rcv_data_ext"))
        if tx is None or rx is None:
            tx = _read_int(os.path.join(cdir, "port_xmit_data"))
            rx = _read_int(os.path.join(cdir, "port_rcv_data"))
            if tx is not None:
                tx *= 4
            if rx is not None:
                rx *= 4
        snap.append({"dev": dev, "port": port, "rate": rate, "tx_bytes": tx or 0, "rx_bytes": rx or 0})
    return snap

def _ib_delta_gbps(s0, s1, elapsed_s):
    """Compute per-port and totals (Gb/s) from two snapshots and elapsed seconds."""
    ports = []
    total_tx_gbps = 0.0
    total_rx_gbps = 0.0
    key = lambda d: (d["dev"], d["port"])
    s0m = {key(d): d for d in s0}
    for d1 in s1:
        k = key(d1)
        d0 = s0m.get(k)
        if not d0:
            continue
        tx_bps = max(0, d1["tx_bytes"] - d0["tx_bytes"]) / elapsed_s
        rx_bps = max(0, d1["rx_bytes"] - d0["rx_bytes"]) / elapsed_s
        tx_gbps = tx_bps * 8 / 1e9
        rx_gbps = rx_bps * 8 / 1e9
        ports.append({"dev": d1["dev"], "port": d1["port"], "rate": d1["rate"],
                      "tx_gbps": tx_gbps, "rx_gbps": rx_gbps})
        total_tx_gbps += tx_gbps
        total_rx_gbps += rx_gbps
    return ports, total_tx_gbps, total_rx_gbps

def _parse_rate_gbps(rate_str):
    try:
        return float(rate_str.split()[0])  # "100 Gb/sec ..." -> 100.0
    except Exception:
        return None

# -------------------- main program --------------------
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ.get("RANK", "0"))
header = f"{socket.gethostname()}-{local_rank}"

if local_rank == 0:
    printflock(f"{header}: torch.__version__: {torch.__version__}")
    printflock(f"{header}: torch.version.cuda: {torch.version.cuda}")
    printflock(f"{header}: torch.cuda.is_available(): {torch.cuda.is_available()}")
    printflock(f"{header}: torch.cuda.nccl.version(): {torch.cuda.nccl.version()}")

printflock(f'{header}: running dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}")) ...')
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
printflock(f'{header}:  dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}")) SUCCESS')

try:
    printflock(f"{header}: Trying dist.barrier()")
    dist.barrier()
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} OK")
    SUCCESS = 1

    # -------------- Configuration --------------
    world_size = dist.get_world_size()
    sizes_env = os.environ.get("NCCL_BW_SIZES_MIB")
    if sizes_env:
        sizes_mib = [int(s.strip()) for s in sizes_env.split(",") if s.strip()]
    else:
        single_env = os.environ.get("NCCL_BW_MIB")
        if single_env:
            sizes_mib = [int(single_env)]
        else:
            sizes_mib = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    iters = int(os.environ.get("NCCL_BW_ITERS", "20"))
    warmup_iters = int(os.environ.get("NCCL_BW_WARMUP", "3"))
    dtype_name = os.environ.get("NCCL_BW_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else (torch.float16 if dtype_name == "fp16" else torch.float32)
    measure_wire = os.environ.get("NCCL_BW_MEASURE_WIRE", "1") != "0"

    # -------------- Topology summary --------------
    nodes, by_node, gpn_map = get_nodes_and_mapping()
    if rank == 0:
        gpn_vals = sorted(set(gpn_map.values()))
        gpn_str = f"{gpn_vals[0]}" if len(gpn_vals) == 1 else "var:" + "/".join(map(str, gpn_vals))
        printflock(f"TOPO: world_size={world_size} nodes={len(nodes)} gpus_per_node={gpn_str} "
                   f"nodelist={nodes}")

    # ---------- Benchmark function (works for any N×G) ----------
    def bench_allreduce(bytes_per_tensor: int):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = max(1, bytes_per_tensor // elem_size)
        tensor = torch.ones(num_elements, dtype=dtype, device=f"cuda:{local_rank}")

        # warmup
        for _ in range(warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        # optional wire snapshot on local node
        if measure_wire and local_rank == 0:
            ib0 = _ib_snapshot()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        total_elapsed = time.perf_counter() - t0
        avg_elapsed = total_elapsed / iters

        if measure_wire and local_rank == 0:
            ib1 = _ib_snapshot()

        # bandwidths
        payload_bw_gBs = bytes_per_tensor / avg_elapsed / 1e9
        bus_bw_gBs = 2 * (world_size - 1) / world_size * payload_bw_gBs

        # collect rank stats (avg latency and both bandwidths)
        stats = {"avg_ms": avg_elapsed * 1e3,
                 "payload_bw_gBs": payload_bw_gBs,
                 "bus_bw_gBs": bus_bw_gBs}
        all_stats = gather_all(stats)

        # Node-local wire printout from local_rank==0
        if measure_wire and local_rank == 0:
            ports, tx_gbps, rx_gbps = _ib_delta_gbps(ib0, ib1, total_elapsed)
            lr_tx = sum((_parse_rate_gbps(p["rate"]) or 0.0) for p in ports)
            lr_rx = lr_tx  # symmetric link rates
            util_tx = 100.0 * tx_gbps / lr_tx if lr_tx > 0 else 0.0
            util_rx = 100.0 * rx_gbps / lr_rx if lr_rx > 0 else 0.0
            host = socket.gethostname()
            per_rail = []
            for p in ports:
                rate_g = _parse_rate_gbps(p["rate"]) or 0.0
                pr_tx = 100.0 * p["tx_gbps"] / rate_g if rate_g > 0 else 0.0
                pr_rx = 100.0 * p["rx_gbps"] / rate_g if rate_g > 0 else 0.0
                per_rail.append(f"{p['dev']}/p{p['port']}: {p['tx_gbps']:.1f}→{p['rx_gbps']:.1f} Gb/s "
                                f"({pr_tx:.0f}%/{pr_rx:.0f}% of {int(rate_g) if rate_g else 0}G)")
            printflock(f"{host}: WIRE size={bytes_per_tensor//(1024*1024)} MiB over {len(ports)} HCAs | "
                       f"TX={tx_gbps:.1f} Gb/s RX={rx_gbps:.1f} Gb/s | util≈ {util_tx:.0f}%/{util_rx:.0f}% ; "
                       f"per-rail: {' | '.join(per_rail)}")

        # rank-0 aggregate view (latency quantiles, bandwidth mins/means)
        if dist.get_rank() == 0:
            avg_ms_vals = [s["avg_ms"] for s in all_stats]
            bus_vals = [s["bus_bw_gBs"] for s in all_stats]
            alg_vals = [s["payload_bw_gBs"] for s in all_stats]

            size_mib = bytes_per_tensor // (1024 * 1024)
            p50 = percentile(avg_ms_vals, 50)
            p90 = percentile(avg_ms_vals, 90)
            p99 = percentile(avg_ms_vals, 99)
            msgps = 1000.0 / p50 if p50 and p50 > 0 else float("nan")

            printflock(
                f"AGG size={size_mib} MiB | lat(ms) p50={p50:.3f} p90={p90:.3f} p99={p99:.3f} | "
                f"busBW GB/s min={min(bus_vals):.2f} mean={sum(bus_vals)/len(bus_vals):.2f} max={max(bus_vals):.2f} | "
                f"algBW GB/s mean={sum(alg_vals)/len(alg_vals):.2f} | msg/s≈{msgps:.2f}"
            )

            # Per-node mean busBW to spot skew
            per_node = []
            for n in nodes:
                idxs = by_node[n]
                node_mean = sum(bus_vals[i] for i in idxs) / max(1, len(idxs))
                per_node.append(f"{n}:{node_mean:.2f}")
            printflock(f"PER-NODE busBW GB/s [{size_mib} MiB] -> " + " ".join(per_node))

        # optional per-rank line (kept compact)
        printflock(f"{header}: AllReduce {bytes_per_tensor//(1024*1024)} MiB "
                   f"avg {stats['avg_ms']:.3f} ms | busBW {bus_bw_gBs:.2f} GB/s | algBW {payload_bw_gBs:.2f} GB/s "
                   f"(iters={iters}, dtype={dtype_name})")

        dist.barrier()
        return avg_elapsed, payload_bw_gBs, bus_bw_gBs

    # -------------------- run sizes --------------------
    for size_mib in sizes_mib:
        if rank == 0:
            printflock("------------------------------------------------------")
            printflock(f"Running AllReduce {size_mib} MiB")
        bytes_per_tensor = size_mib * 1024 * 1024
        bench_allreduce(bytes_per_tensor)
    if rank == 0:
        printflock("------------------------------------------------------")

except Exception as e:
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} ERROR: {e}")
    raise
finally:
    if dist.is_initialized():
        printflock(f"{header}: Destroying process group...")
        dist.destroy_process_group()
        printflock(f"{header}: Process group destroyed successfully")
    time.sleep(1)
    printflock(f"{header}: NCCL TEST SUCCESS: {bool(SUCCESS)}")
EOT


echo "============================="
echo "Software versions:"
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): nvidia-smi: $(nvidia-smi)"'
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): nvidia driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"'
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): nvcc version: $(nvcc --version)"'
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): ibstat: $(ibstat)"'
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): ibdev2netdev: $(ibdev2netdev)"'
srun --jobid $SLURM_JOBID bash -c 'echo "$(hostname): ofed_info -s: $(ofed_info -s)"'
srun --jobid $SLURM_JOBID bash -c "mods=\$(lsmod | grep -E 'nvidia_peermem|nv_peer_mem' || true); echo \"\$(hostname): lsmod grep -E nvidia_peermem|nv_peer_mem: \$mods\""

echo "============================="
echo "NCCL env vars:"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "NCCL_IB_HCA: $NCCL_IB_HCA"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING: $TORCH_NCCL_ASYNC_ERROR_HANDLING"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "NCCL_NET_GDR_READ: $NCCL_NET_GDR_READ"
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "TORCH_NCCL_BLOCKING_WAIT: $TORCH_NCCL_BLOCKING_WAIT"
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: $TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
echo "TORCH_COMPILE_DISABLE: $TORCH_COMPILE_DISABLE"
echo "ACCELERATE_USE_CUDA_GRAPHS: $ACCELERATE_USE_CUDA_GRAPHS"
echo "TORCH_CUDAGRAPHS: $TORCH_CUDAGRAPHS"
echo "TORCHDYNAMO_DISABLE: $TORCHDYNAMO_DISABLE"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_NET: $NCCL_NET"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NCCL_SHM_DISABLE: $NCCL_SHM_DISABLE"
echo "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"


echo "============================="
echo "Running NCCL test:"
echo "NCCL launcher:" "${LAUNCHER[@]}" --node_rank "$SLURM_PROCID" "$SCRIPT"
SRUN_NCCL=(
    srun -u --jobid "$SLURM_JOBID" --time 10:00 --kill-on-bad-exit=1 --wait=60
)
"${SRUN_NCCL[@]}" "${LAUNCHER[@]}" --node_rank "$SLURM_NODEID" "$SCRIPT"
nccl_rc=$?
if [ $nccl_rc -ne 0 ]; then
    echo "NCCL test failed with rc=$nccl_rc"
    exit $nccl_rc
fi
