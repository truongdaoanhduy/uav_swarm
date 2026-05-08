# eval_and_upload.py
"""
Script eval 3 algo và upload kết quả lên HuggingFace.

Usage:
    # Checkpoint trên HuggingFace (download về rồi eval)
    python eval_and_upload.py \
        --hf-token    "hf_xxxxxxxxxxxx" \
        --hf-repo     "username/sar-uav-results" \
        --stages      hard extreme transfer \
        --n-episodes  100 \
        --n-envs      4 \
        --device      cpu

    # Checkpoint local (đã có sẵn trên máy/Kaggle)
    python eval_and_upload.py \
        --hf-token    "hf_xxxxxxxxxxxx" \
        --hf-repo     "username/sar-uav-results" \
        --mappo       "mappo_s42/checkpoint_final.pt" \
        --masac       "masac_s42/checkpoint_final.pt" \
        --matd3       "matd3_s42/checkpoint_final.pt" \
        --stages      hard extreme transfer \
        --n-episodes  100 \
        --n-envs      4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# JSON HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """JSON serializer cho numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD CHECKPOINT TỪ HUGGINGFACE
# ══════════════════════════════════════════════════════════════════════════════

def download_checkpoints_from_hf(
    hf_token:  str,
    hf_repo:   str,
    local_dir: str,
    algos:     List[str] = ["mappo", "masac", "matd3"],
) -> Dict[str, str]:
    """
    Download checkpoint từ HuggingFace về local.

    Tự động tìm file .pt trong repo theo nhiều pattern:
        mappo_s42/checkpoint_final.pt
        mappo/checkpoint_final.pt
        checkpoints/mappo/checkpoint_final.pt
        ...

    Returns:
        Dict[algo -> local_path]
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("❌ pip install huggingface-hub")
        return {}

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # List tất cả files trong repo 1 lần
    print(f"\n  🔍 Listing files trong HF repo: {hf_repo}")
    try:
        all_files = list(list_repo_files(
            hf_repo,
            token     = hf_token,
            repo_type = "dataset",
        ))
        # In ra để debug
        pt_files = [f for f in all_files if f.endswith(".pt")]
        print(f"  📋 Tìm thấy {len(pt_files)} file .pt:")
        for f in pt_files:
            print(f"       {f}")
    except Exception as e:
        print(f"  ❌ Không list được files: {e}")
        all_files = []

    downloaded = {}

    for algo in algos:
        print(f"\n  📥 Tìm checkpoint cho {algo.upper()}...")

        # Tìm file .pt chứa tên algo trong danh sách
        algo_pt_files = [
            f for f in all_files
            if algo in f.lower() and f.endswith(".pt")
        ]

        if not algo_pt_files:
            print(f"  ❌ {algo}: Không tìm thấy .pt file nào trong repo")
            continue

        # Ưu tiên checkpoint_final, sau đó lấy file cuối cùng
        chosen = None
        for f in algo_pt_files:
            if "final" in f:
                chosen = f
                break
        if chosen is None:
            chosen = algo_pt_files[-1]

        print(f"  ⬇️  Downloading: {chosen}")
        try:
            local_path = hf_hub_download(
                repo_id   = hf_repo,
                filename  = chosen,
                token     = hf_token,
                repo_type = "dataset",
                local_dir = str(local_dir / algo),
            )
            downloaded[algo] = local_path
            size_mb = Path(local_path).stat().st_size / 1e6
            print(f"  ✅ {algo.upper()}: {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ {algo}: Download thất bại: {e}")

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOADER
# ══════════════════════════════════════════════════════════════════════════════

class EvalHFUploader:
    """Upload kết quả evaluation JSON lên HuggingFace."""

    def __init__(self, token: str, repo_id: str):
        self.token   = token
        self.repo_id = repo_id
        self._api    = None
        self._init()

    def _init(self):
        try:
            from huggingface_hub import HfApi, create_repo
            self._api = HfApi(token=self.token)
            create_repo(
                self.repo_id,
                token     = self.token,
                exist_ok  = True,
                repo_type = "dataset",
            )
            print(f"  ✅ HF repo sẵn sàng: {self.repo_id}")
        except ImportError:
            print("  ❌ pip install huggingface-hub")
        except Exception as e:
            print(f"  ⚠️  HF repo init: {e}")

    def upload_file(
        self,
        local_path:  str,
        repo_path:   str,
        commit_msg:  str = "Upload eval results",
    ) -> bool:
        if self._api is None:
            return False
        try:
            self._api.upload_file(
                path_or_fileobj = local_path,
                path_in_repo    = repo_path,
                repo_id         = self.repo_id,
                repo_type       = "dataset",
                commit_message  = commit_msg,
            )
            print(f"  ☁️  Uploaded → {repo_path}")
            return True
        except Exception as e:
            print(f"  ❌ Upload thất bại {repo_path}: {e}")
            return False

    def upload_stage_results(
        self,
        stage:        str,
        algo_results: Dict,
        comparison:   Optional[Dict],
        stage_dir:    Path,
    ):
        """Upload tất cả JSON của 1 stage."""
        # Từng algo
        for algo, result in algo_results.items():
            local_path = stage_dir / f"{algo}_{stage}.json"
            result_dict = (
                asdict(result)
                if hasattr(result, "__dataclass_fields__")
                else result
            )
            with open(local_path, "w") as f:
                json.dump(result_dict, f, indent=2, default=_json_safe)
            self.upload_file(
                str(local_path),
                f"eval_results/{stage}/{algo}_results.json",
                f"Eval {algo.upper()} on {stage}",
            )

        # Comparison
        if comparison:
            comp_path = stage_dir / f"comparison_{stage}.json"
            with open(comp_path, "w") as f:
                json.dump(comparison, f, indent=2, default=_json_safe)
            self.upload_file(
                str(comp_path),
                f"eval_results/{stage}/comparison.json",
                f"Statistical comparison {stage}",
            )

    def upload_summary(
        self,
        all_stage_results: Dict,
        out_dir: Path,
    ) -> Dict:
        """Tổng hợp tất cả stages → summary.json → upload."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "stages":    {},
            "rankings":  {},
            "cross_stage_analysis": {},
        }

        for stage, algo_results in all_stage_results.items():
            summary["stages"][stage] = {}
            for algo, r in algo_results.items():
                if hasattr(r, "reward_mean"):
                    summary["stages"][stage][algo] = {
                        "reward_mean":      float(r.reward_mean),
                        "reward_std":       float(r.reward_std),
                        "reward_median":    float(r.reward_median),
                        "coverage_mean":    float(r.coverage_mean),
                        "coverage_std":     float(r.coverage_std),
                        "victim_rate_mean": float(r.victim_rate_mean),
                        "victim_rate_std":  float(r.victim_rate_std),
                        "success_rate":     float(r.success_rate),
                        "n_episodes":       int(r.n_episodes),
                        "checkpoint":       str(r.checkpoint),
                    }

            if algo_results:
                summary["rankings"][stage] = {
                    "by_reward": sorted(
                        algo_results,
                        key=lambda a: float(
                            getattr(algo_results[a], "reward_mean", 0)
                        ),
                        reverse=True,
                    ),
                    "by_coverage": sorted(
                        algo_results,
                        key=lambda a: float(
                            getattr(algo_results[a], "coverage_mean", 0)
                        ),
                        reverse=True,
                    ),
                    "by_success": sorted(
                        algo_results,
                        key=lambda a: float(
                            getattr(algo_results[a], "success_rate", 0)
                        ),
                        reverse=True,
                    ),
                }

        # Cross-stage generalization gap
        if "hard" in all_stage_results and "extreme" in all_stage_results:
            hard_r    = all_stage_results["hard"]
            extreme_r = all_stage_results["extreme"]
            for algo in set(hard_r) & set(extreme_r):
                h_cov = float(getattr(hard_r[algo],    "coverage_mean", 0))
                e_cov = float(getattr(extreme_r[algo], "coverage_mean", 0))
                h_rew = float(getattr(hard_r[algo],    "reward_mean",   0))
                e_rew = float(getattr(extreme_r[algo], "reward_mean",   0))
                summary["cross_stage_analysis"][algo] = {
                    "generalization_gap_coverage": round(h_cov - e_cov, 4),
                    "generalization_gap_reward":   round(h_rew - e_rew, 2),
                    "coverage_retention_pct":      round(
                        e_cov / max(h_cov, 1e-8) * 100, 1
                    ),
                }

        # Print
        self._print_summary(summary)

        # Save + upload
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_safe)
        self.upload_file(
            str(summary_path),
            "eval_results/summary.json",
            "Overall evaluation summary",
        )

        return summary

    @staticmethod
    def _print_summary(summary: Dict):
        print(f"\n{'═'*65}")
        print(f"  📊 EVALUATION SUMMARY")
        print(f"{'═'*65}")
        for stage, algos in summary.get("stages", {}).items():
            print(f"\n  Stage: {stage.upper()}")
            print(f"  {'Algo':<10} {'Reward':>10} "
                  f"{'Coverage':>10} {'VictimRate':>12} {'Success':>10}")
            print(f"  {'-'*54}")
            for algo, m in algos.items():
                print(
                    f"  {algo.upper():<10} "
                    f"{m['reward_mean']:>10.1f} "
                    f"{m['coverage_mean']*100:>9.1f}% "
                    f"{m['victim_rate_mean']*100:>11.1f}% "
                    f"{m['success_rate']*100:>9.1f}%"
                )
        if summary.get("cross_stage_analysis"):
            print(f"\n  Generalization Gap (HARD → EXTREME):")
            for algo, g in summary["cross_stage_analysis"].items():
                print(
                    f"    {algo.upper():<8}: "
                    f"coverage retention = {g['coverage_retention_pct']:.1f}%  "
                    f"(gap = {g['generalization_gap_coverage']:+.3f})"
                )
        print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    hf_token:    str,
    hf_repo:     str,
    checkpoints: Optional[Dict[str, str]] = None,
    stages:      List[str]                = ["hard", "extreme", "transfer"],
    n_episodes:  int                      = 100,
    base_seed:   int                      = 9999,
    device:      str                      = "cpu",
    n_envs:      int                      = 1,
    n_uav:       int                      = 4,
    output_dir:  str                      = "results/eval",
) -> Dict:
    """
    Eval 3 algo trên nhiều stages → lưu JSON → upload HF.
    """
    from evaluate import build_eval_config, evaluate_algo, compare_algos

    print(f"\n{'═'*65}")
    print(f"  🚁 SAR UAV SWARM — FULL EVALUATION PIPELINE")
    print(f"{'═'*65}")
    print(f"  Stages     : {stages}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Parallel   : {n_envs} envs")
    print(f"  Device     : {device}")
    print(f"  HF Repo    : {hf_repo}")
    print(f"{'═'*65}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uploader = EvalHFUploader(hf_token, hf_repo)

    # ── Download nếu chưa có checkpoint local ─────────────────────────────────
    if not checkpoints:
        print("\n  📥 Không có checkpoint local → Download từ HF...")
        checkpoints = download_checkpoints_from_hf(
            hf_token  = hf_token,
            hf_repo   = hf_repo,
            local_dir = str(out_dir / "ckpts"),
        )

    if not checkpoints:
        print("  ❌ Không có checkpoint nào để eval!")
        return {}

    print(f"\n  📋 Checkpoint sẽ dùng:")
    for algo, path in checkpoints.items():
        size_mb = Path(path).stat().st_size / 1e6 if Path(path).exists() else 0
        print(f"    {algo.upper():<8}: {Path(path).name} ({size_mb:.1f} MB)")

    # ── Eval từng stage ───────────────────────────────────────────────────────
    all_stage_results = {}

    for stage in stages:
        print(f"\n{'─'*65}")
        print(f"  📍 STAGE: {stage.upper()}")
        print(f"{'─'*65}")

        stage_dir = out_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        cfg          = build_eval_config(stage, n_uav)
        algo_results = {}

        for algo, ckpt_path in checkpoints.items():
            if not Path(ckpt_path).exists():
                print(f"  ⚠️  {algo}: file không tồn tại: {ckpt_path}")
                continue

            print(f"\n  🔄 {algo.upper()} on {stage.upper()}...")
            t0 = time.time()

            try:
                result = evaluate_algo(
                    algo            = algo,
                    checkpoint_path = ckpt_path,
                    config          = cfg,
                    stage           = stage,
                    n_episodes      = n_episodes,
                    base_seed       = base_seed,
                    device          = device,
                    n_envs          = n_envs,
                    verbose         = True,
                )
                algo_results[algo] = result
                print(f"  ⏱️  {algo.upper()} done in {time.time()-t0:.1f}s")

            except Exception as e:
                import traceback
                print(f"  ❌ {algo.upper()} failed: {e}")
                traceback.print_exc()

        all_stage_results[stage] = algo_results

        # Statistical comparison
        comparison = None
        if len(algo_results) >= 2:
            print(f"\n  📊 Statistical comparison ({stage.upper()})...")
            try:
                comparison = compare_algos(
                    results     = algo_results,
                    output_path = str(stage_dir / f"comparison_{stage}.json"),
                    verbose     = True,
                )
            except Exception as e:
                print(f"  ⚠️  Compare failed: {e}")

        # Upload stage
        uploader.upload_stage_results(stage, algo_results, comparison, stage_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = uploader.upload_summary(all_stage_results, out_dir)

    # Full dump
    full = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "config":      {"stages": stages, "n_episodes": n_episodes,
                        "n_envs": n_envs, "device": device},
        "checkpoints": {a: str(p) for a, p in checkpoints.items()},
        "summary":     summary,
    }
    full_path = out_dir / "full_eval_results.json"
    with open(full_path, "w") as f:
        json.dump(full, f, indent=2, default=_json_safe)
    uploader.upload_file(
        str(full_path),
        "eval_results/full_eval_results.json",
        "Full evaluation results",
    )

    print(f"\n{'═'*65}")
    print(f"  ✅ DONE!")
    print(f"  📁 Local : {out_dir}")
    print(f"  ☁️  HF    : https://huggingface.co/datasets/{hf_repo}/tree/main/eval_results")
    print(f"{'═'*65}\n")

    return {"algo_results": all_stage_results, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Eval MAPPO/MASAC/MATD3 và upload lên HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── HuggingFace ──────────────────────────────────────────────────────────
    p.add_argument(
        "--hf-token",
        type    = str,
        default = None,
        help    = "HuggingFace API token (hoặc set env HF_TOKEN)",
    )
    p.add_argument(
        "--hf-repo",
        type     = str,
        required = True,
        help     = "HF repo ID để UPLOAD kết quả, vd: username/sar-uav-eval",
    )
    p.add_argument(
        "--hf-ckpt-repo",
        type    = str,
        default = None,
        help    = (
            "HF repo ID chứa CHECKPOINT (nếu khác --hf-repo). "
            "Nếu không truyền → dùng chung --hf-repo."
        ),
    )

    # ── Checkpoint local (optional, ưu tiên hơn download) ────────────────────
    p.add_argument("--mappo", type=str, default=None,
                   help="Path local đến MAPPO checkpoint_final.pt")
    p.add_argument("--masac", type=str, default=None,
                   help="Path local đến MASAC checkpoint_final.pt")
    p.add_argument("--matd3", type=str, default=None,
                   help="Path local đến MATD3 checkpoint_final.pt")

    # ── Eval params ───────────────────────────────────────────────────────────
    p.add_argument(
        "--stages",
        nargs   = "+",
        default = ["hard", "extreme", "transfer"],
        choices = ["hard", "extreme", "transfer"],
        help    = "Stages để eval",
    )
    p.add_argument("--n-episodes", type=int, default=100,
                   help="Số episodes mỗi stage")
    p.add_argument("--n-envs",    type=int, default=1,
                   help="Số parallel envs (1=sequential, 4=4× faster)")
    p.add_argument("--seed",      type=int, default=9999)
    p.add_argument("--device",    type=str, default="cpu")
    p.add_argument("--n-uav",     type=int, default=4)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        type    = str,
        default = "results/eval",
        help    = "Thư mục lưu kết quả JSON local",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── Lấy HF token ─────────────────────────────────────────────────────────
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not hf_token:
        print("❌ Cần HF token!\n"
              "   Truyền: --hf-token hf_xxxx\n"
              "   Hoặc:   export HF_TOKEN=hf_xxxx")
        return

    # Repo chứa checkpoint (có thể khác repo upload)
    ckpt_repo = args.hf_ckpt_repo or args.hf_repo

    # ── Build checkpoints dict ────────────────────────────────────────────────
    checkpoints: Dict[str, str] = {}

    for algo, path in [
        ("mappo", args.mappo),
        ("masac", args.masac),
        ("matd3", args.matd3),
    ]:
        if path is None:
            continue
        p = Path(path)
        if p.exists():
            checkpoints[algo] = str(p)
            print(f"  ✅ {algo.upper()}: dùng local checkpoint: {p}")
        else:
            print(f"  ⚠️  {algo.upper()}: file không tồn tại: {path}")

    # Nếu thiếu → download từ HF checkpoint repo
    missing = [a for a in ["mappo", "masac", "matd3"]
               if a not in checkpoints]
    if missing:
        print(f"\n  📥 Download checkpoint còn thiếu {missing} từ: {ckpt_repo}")
        downloaded = download_checkpoints_from_hf(
            hf_token  = hf_token,
            hf_repo   = ckpt_repo,
            local_dir = str(Path(args.output_dir) / "ckpts"),
            algos     = missing,
        )
        checkpoints.update(downloaded)

    if not checkpoints:
        print("  ❌ Không có checkpoint nào!")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    run_full_evaluation(
        hf_token    = hf_token,
        hf_repo     = args.hf_repo,
        checkpoints = checkpoints,
        stages      = args.stages,
        n_episodes  = args.n_episodes,
        base_seed   = args.seed,
        device      = args.device,
        n_envs      = args.n_envs,
        n_uav       = args.n_uav,
        output_dir  = args.output_dir,
    )


if __name__ == "__main__":
    main()