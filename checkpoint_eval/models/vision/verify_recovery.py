"""Correctness harness for pccheck checkpoint/recovery.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.multiprocessing as mp
from torchvision import models

from checkpoint_eval.pccheck_utils import initialize_updated, set_storage_updated

# Determinism: must be set before any CUDA context work.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------- fingerprints


def _hash(t):
    """Stable content hash of a tensor, independent of device and layout."""
    return hashlib.sha256(
        t.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()[:16]


def fingerprint(model, optimizer, gpu_ar):
    """Component-wise hashes, so a mismatch says *which* state went wrong.

    Kept separate rather than hashing gpu_ar alone: params, BN buffers and
    optimizer momentum live in different places, and only some of them are
    inside the flat buffer (see the `audit` subcommand).
    """
    params = {n: _hash(p) for n, p in model.named_parameters()}

    buffers = {n: _hash(b) for n, b in model.named_buffers()}

    momentum = {}
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            st = optimizer.state.get(p, {})
            if "momentum_buffer" in st and st["momentum_buffer"] is not None:
                momentum[f"g{gi}.p{pi}"] = _hash(st["momentum_buffer"])

    grads = {
        f"g{gi}.p{pi}": _hash(p.grad)
        for gi, group in enumerate(optimizer.param_groups)
        for pi, p in enumerate(group["params"])
        if p.grad is not None
    }

    return {
        "flat_buffer": _hash(gpu_ar),
        "params": _hash_dict(params),
        "buffers": _hash_dict(buffers),
        "momentum": _hash_dict(momentum),
        "grads": _hash_dict(grads),
        "_detail": {
            "params": params,
            "buffers": buffers,
            "momentum": momentum,
            "grads": grads,
        },
    }


def _hash_dict(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------- setup


def build(args):
    """Construct model/optimizer exactly as train_pccheck.py does."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Must precede any Chk_monitor: it starts the checkpoint worker as a
    # multiprocessing.Process, and the default 'fork' would copy a process that
    # already holds a CUDA context. CUDA is unusable in such a child, so
    # cudaMallocHost in DRAMAlloc::alloc fails its assert, the worker dies
    # before reaching its barrier, and the parent blocks in barrier.wait()
    # forever. Same call train_pccheck.py makes.
    mp.set_start_method("spawn", force=True)

    torch.cuda.set_device(0)
    num_classes = 1000 if args.dataset == "imagenet" else 10
    model = models.__dict__[args.arch](num_classes=num_classes).to(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
    metric_fn = torch.nn.CrossEntropyLoss().to(0)
    model.train()

    # Updated flattening: parameters, optimizer state and model buffers all
    # get re-pointed into one contiguous GPU buffer, which is what pccheck
    # persists. The original set_storage() is still in pccheck_utils for the
    # models that have not been migrated yet.
    gpu_ar, total_size = initialize_updated(model, [optimizer])
    set_storage_updated(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()

    _reset_to_pristine(args, model, optimizer, num_classes)
    torch.cuda.empty_cache()

    return model, optimizer, metric_fn, gpu_ar, total_size


def _reset_to_pristine(args, model, optimizer, num_classes):
    """Undo the damage initialize() does to the weights.

    pccheck_utils.initialize() allocates p.grad from *uninitialized* memory
    (p.data.new(...)) and then calls optimizer.step(), which applies that
    garbage to the weights -- the model is nan/garbage before training starts.
    train_pccheck.py never notices because it only times checkpoints, but a
    correctness harness needs a model that actually trains.

    We overwrite in place (copy_, not assignment) so params keep aliasing the
    flat buffer -- the layout under test must not change.
    """
    torch.manual_seed(args.seed)
    pristine = models.__dict__[args.arch](num_classes=num_classes)

    with torch.no_grad():
        for (_, p), (_, q) in zip(model.named_parameters(),
                                  pristine.named_parameters()):
            p.copy_(q.to(p.device))
        for (_, b), (_, c) in zip(model.named_buffers(),
                                  pristine.named_buffers()):
            b.copy_(c.to(b.device))

        # start from a clean optimizer/grad state rather than garbage
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()
                st = optimizer.state.get(p, {})
                if torch.is_tensor(st.get("momentum_buffer")):
                    st["momentum_buffer"].zero_()

    del pristine


def get_batch(args, step):
    """Deterministic per-step batch.

    Seeded by step (not by iteration order) so that a resumed run at step k
    sees byte-identical data to the reference run at step k. Random rather
    than torch.ones so the loss actually moves and can expose bad state.
    """
    g = torch.Generator().manual_seed(args.seed * 1_000_003 + step)
    num_classes = 1000 if args.dataset == "imagenet" else 10
    data = torch.randn(args.batchsize, 3, 224, 224, generator=g)
    target = torch.randint(num_classes, (args.batchsize,), generator=g)
    return data.to(0, non_blocking=False), target.to(0, non_blocking=False)


def train_step(args, model, optimizer, metric_fn, step):
    optimizer.zero_grad()
    data, target = get_batch(args, step)
    output = model(data)
    if args.arch == "inception_v3":
        output = output[0]
    loss = metric_fn(output, target)
    loss.backward()
    optimizer.step()

    val = loss.item()
    if val != val or val in (float("inf"), float("-inf")):
        # a nan/inf trajectory compares equal to itself and would make the
        # whole trajectory check vacuous -- refuse to produce such a trace
        raise RuntimeError(
            f"loss is {val} at step {step}. The comparison would be "
            f"meaningless. Lower --lr (currently {args.lr}) or check that "
            f"_reset_to_pristine() ran."
        )
    return val


# ------------------------------------------------------------------- backends


class TorchBackend:
    """Oracle backend. Saves the same state pccheck would, via torch.save."""

    def __init__(self, args):
        self.path = args.ckpt_path + ".torch"

    def save(self, model, optimizer, gpu_ar, step):
        # Since set_storage() maps parameters, optimizer state and model
        # buffers all into gpu_ar, the flat buffer alone is the complete
        # training state -- nothing needs saving alongside it.
        torch.save({"step": step, "flat_buffer": gpu_ar.detach().cpu()},
                   self.path)

    def load(self, model, optimizer, gpu_ar):
        ckpt = torch.load(self.path, map_location="cuda:0", weights_only=False)
        # One copy restores everything, because every destination is a view of
        # gpu_ar. Deliberately NOT optimizer.load_state_dict(): that replaces
        # the momentum tensors with fresh allocations, silently breaking their
        # aliasing to the flat buffer and leaving gpu_ar's momentum region
        # stale from then on.
        gpu_ar.copy_(ckpt["flat_buffer"].to(0))
        return ckpt["step"]


class PcCheckBackend:
    """Real pccheck path. Save works today; load is steps 2 and 3."""

    def __init__(self, args):
        self.args = args
        self.monitor = None

    def save(self, model, optimizer, gpu_ar, step):
        from checkpoint_eval.pccheck.chk_monitor import Chk_monitor

        if self.monitor is None:
            self.monitor = Chk_monitor(
                self.args.c_lib_path,
                gpu_ar.numel(),
                self.args.num_threads,
                self.args.max_async,
                True,
                gpu_ar=gpu_ar,
                bsize=gpu_ar.numel(),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                memory_saving=True,
            )
        self.monitor.save()

        # Block until the worker has finished copying gpu_ar into its staging
        # buffer. Checkpointing is asynchronous, so without this the next
        # optimizer.step() would mutate the buffer mid-copy and persist a torn
        # mixture of two steps -- which would show up here as a bogus
        # correctness failure rather than as the race it is.
        while self.monitor.gpu_copy_in_progress():
            pass

    def close(self):
        if self.monitor is not None:
            self.monitor.kill_checkpoint()
            self.monitor = None

    def load(self, model, optimizer, gpu_ar):
        # Read before any Chk_monitor/writer exists: initialize() has side
        # effects on the head pointer and the free-slot queue, and loading
        # must not depend on them.
        from checkpoint_eval.pccheck.chk_load import load_flat, restore_from_flat

        flat, info = load_flat(self.args.ckpt_path, gpu_ar.numel(),
                               self.args.max_async)
        report = restore_from_flat(model, [optimizer], flat)
        print(f"[pccheck] loaded slot {info['slot']} counter {info['counter']} "
              f"from byte {info['byte_offset']}")
        if "warning" in report:
            print(f"[pccheck] WARNING: {report['warning']}")

        # The counter is a checkpoint sequence number, not a training step.
        # Until the step is persisted alongside the payload (step 3), fall
        # back to the step the caller asked us to resume from.
        return self.args.ckpt_step


def make_backend(args):
    return {"torch": TorchBackend, "pccheck": PcCheckBackend}[args.backend](args)


# ------------------------------------------------------------------ subcommands


def cmd_reference(args):
    """Uninterrupted run. Checkpoints at --ckpt-step, keeps going to --steps."""
    model, optimizer, metric_fn, gpu_ar, total_size = build(args)
    backend = make_backend(args)

    trace = []
    for step in range(args.steps):
        loss = train_step(args, model, optimizer, metric_fn, step)
        trace.append({"step": step, "loss": repr(loss)})

        if step == args.ckpt_step:
            backend.save(model, optimizer, gpu_ar, step)
            trace[-1]["checkpointed"] = True
            trace[-1]["state"] = fingerprint(model, optimizer, gpu_ar)

    final = fingerprint(model, optimizer, gpu_ar)
    if hasattr(backend, "close"):
        backend.close()

    _write(args.out, {
        "mode": "reference", "total_size": total_size,
        "args": _argsdict(args), "trace": trace, "final": final,
    })


def cmd_resume(args):
    """Fresh process: load the checkpoint, run the remaining steps."""
    model, optimizer, metric_fn, gpu_ar, total_size = build(args)
    backend = make_backend(args)

    loaded_step = backend.load(model, optimizer, gpu_ar)
    restored = fingerprint(model, optimizer, gpu_ar)

    trace = []
    for step in range(loaded_step + 1, args.steps):
        loss = train_step(args, model, optimizer, metric_fn, step)
        trace.append({"step": step, "loss": repr(loss)})

    final = fingerprint(model, optimizer, gpu_ar)
    if hasattr(backend, "close"):
        backend.close()

    _write(args.out, {
        "mode": "resume", "total_size": total_size, "loaded_step": loaded_step,
        "args": _argsdict(args), "restored": restored,
        "trace": trace, "final": final,
    })


def cmd_compare(args):
    ref = json.load(open(args.ref))
    res = json.load(open(args.res))

    ok = True
    print(f"reference: {len(ref['trace'])} steps   resume: {len(res['trace'])} steps"
          f"   (resumed at step {res.get('loaded_step')})")

    # 1. state at the moment of restore vs state when the checkpoint was taken
    saved = next((t["state"] for t in ref["trace"] if t.get("checkpointed")), None)
    if saved and "restored" in res:
        print("\n[restore] checkpointed state vs restored state")
        for comp in ("flat_buffer", "params", "buffers", "momentum", "grads"):
            match = saved[comp] == res["restored"][comp]
            # grads are recomputed by the next backward() before anything reads
            # them, so they need not survive a restore -- report, don't fail.
            if comp != "grads":
                ok &= match
            note = "" if comp != "grads" else "  (informational)"
            print(f"  {comp:12s} {'MATCH' if match else 'MISMATCH'}"
                  f"  {saved[comp]} vs {res['restored'][comp]}{note}")
            if not match and comp != "grads":
                _detail(saved, res["restored"], comp)

    # 2. loss trajectory after the resume point
    ref_loss = {t["step"]: t["loss"] for t in ref["trace"]}
    print("\n[trajectory] losses after resume")
    diverged = None
    for t in res["trace"]:
        r = ref_loss.get(t["step"])
        if r != t["loss"]:
            diverged = t["step"] if diverged is None else diverged
            ok = False
    if diverged is None:
        print(f"  {'step':>5} {'reference':>18} {'resumed':>18}   (bit-identical)")
        for t in res["trace"]:
            print(f"  {t['step']:5d} {float(ref_loss[t['step']]):18.12f} "
                  f"{float(t['loss']):18.12f}")
    else:
        print(f"  FIRST DIVERGENCE at step {diverged}")
        for t in res["trace"]:
            if t["step"] >= diverged - 1:
                mark = "  " if ref_loss.get(t["step"]) == t["loss"] else "<-"
                print(f"    step {t['step']:4d} ref={ref_loss.get(t['step'])} "
                      f"resume={t['loss']} {mark}")

    # 3. final state
    print("\n[final] end-of-run state")
    for comp in ("flat_buffer", "params", "buffers", "momentum", "grads"):
        match = ref["final"][comp] == res["final"][comp]
        ok &= match
        print(f"  {comp:12s} {'MATCH' if match else 'MISMATCH'}")
        if not match:
            _detail(ref["final"], res["final"], comp)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _detail(a, b, comp, limit=5):
    da, db = a.get("_detail", {}).get(comp, {}), b.get("_detail", {}).get(comp, {})
    bad = [k for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
    for k in bad[:limit]:
        print(f"      {k}: {da.get(k)} vs {db.get(k)}")
    if len(bad) > limit:
        print(f"      ... and {len(bad) - limit} more")
    if not bad and (set(da) != set(db)):
        print(f"      key sets differ: {len(da)} vs {len(db)} entries")


def cmd_audit(args):
    """Report which state tensors actually live inside the flat buffer.

    This is the ground truth for what a gpu_ar-only restore can possibly
    recover -- anything listed as OUTSIDE must be persisted some other way.
    """
    model, optimizer, _, gpu_ar, total_size = build(args)

    base = gpu_ar.data_ptr()
    end = base + gpu_ar.numel() * gpu_ar.element_size()

    def where(t):
        p = t.data_ptr()
        return "inside " if base <= p < end else "OUTSIDE"

    print(f"flat buffer: {total_size} floats "
          f"({total_size * 4 / 1e6:.1f} MB), [{hex(base)}, {hex(end)})\n")

    covered = 0
    print("model parameters:")
    for n, p in model.named_parameters():
        loc = where(p)
        covered += p.numel() if loc == "inside " else 0
        print(f"  {loc}  {n:50s} {p.numel():>10d}")

    print("\nmodel buffers (BN running stats etc):")
    nb = 0
    for n, b in model.named_buffers():
        nb += 1
        print(f"  {where(b)}  {n:50s} {b.numel():>10d}")
    if nb == 0:
        print("  (none)")

    print("\noptimizer state:")
    ns = 0
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            for k, v in optimizer.state.get(p, {}).items():
                if torch.is_tensor(v):
                    ns += 1
                    print(f"  {where(v)}  g{gi}.p{pi}.{k:42s} {v.numel():>10d}")
    if ns == 0:
        print("  (none)")

    print("\ngradients:")
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            if p.grad is not None:
                print(f"  {where(p.grad)}  g{gi}.p{pi}.grad{' ':38s}"
                      f"{p.grad.numel():>10d}")

    print(f"\nsummary: {covered}/{total_size} floats of the buffer are model "
          f"parameters; the remainder is whatever set_storage() mapped next.")


# ----------------------------------------------------------------------- misc


def _argsdict(args):
    return {k: v for k, v in vars(args).items() if k != "func"}


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--arch", default="resnet18")
        sp.add_argument("--dataset", default="cifar")
        sp.add_argument("--batchsize", default=32, type=int)
        sp.add_argument("--seed", default=1234, type=int)
        sp.add_argument("--lr", default=0.01, type=float)
        sp.add_argument("--steps", default=20, type=int)
        sp.add_argument("--ckpt-step", default=10, type=int)
        sp.add_argument("--backend", default="torch", choices=["torch", "pccheck"])
        sp.add_argument("--ckpt-path",
                        default="/local/home/fstrati/pccheck_checkpoint.chk")
        sp.add_argument("--c_lib_path", default="")
        sp.add_argument("--num-threads", dest="num_threads", default=2, type=int)
        sp.add_argument("--max-async", dest="max_async", default=2, type=int)

    sp = sub.add_parser("reference"); common(sp)
    sp.add_argument("--out", default="/tmp/pccheck_ref.json")
    sp.set_defaults(func=cmd_reference)

    sp = sub.add_parser("resume"); common(sp)
    sp.add_argument("--out", default="/tmp/pccheck_res.json")
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("audit"); common(sp)
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("compare")
    sp.add_argument("ref"); sp.add_argument("res")
    sp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    os.sched_setaffinity(0, {0})
    main()
