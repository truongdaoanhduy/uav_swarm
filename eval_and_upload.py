"""
Script eval 3 algo trên EXTREME + TRANSFER và upload lên HuggingFace.

EXTREME:  Zero-shot eval (khó hơn HARD - nơi đã train)
TRANSFER: Cross-domain eval (môi trường khác)

Usage:
    python eval_and_upload.py \
        --hf-token    "hf_xxxxxxxxxxxx" \
        --hf-repo     "username/sar-uav-results" \
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

# eval_and_upload.py - THÊM vào đầu file nếu chưa có

def _json_safe(obj):
    """JSON serializer cho numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, bool):  # ← THÊM: Handle bool TRƯỚC int
        return bool(obj)
    return obj

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
        """Initialize HF API và tạo repo nếu chưa có."""
        try:
            from huggingface_hub import HfApi, create_repo
            
            self._api = HfApi(token=self.token)
            
            create_repo(
                self.repo_id,
                token     = self.token,
                exist_ok  = True,
                repo_type = "dataset",
                private   = False,
            )
            print(f"  ✅ HF repo sẵn sàng: {self.repo_id}")
            
        except ImportError:
            print("  ❌ pip install huggingface-hub")
            self._api = None
            
        except Exception as e:
            print(f"  ⚠️  HF repo init failed: {e}")
            self._api = None

    def upload_file(
        self,
        local_path:  str,
        repo_path:   str,
        commit_msg:  str = "Upload eval results",
    ) -> bool:
        """Upload 1 file lên HF repo."""
        if self._api is None:
            print(f"  ⚠️  HF API not initialized, skip upload: {repo_path}")
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
            print(f"  ❌ Upload failed {repo_path}: {e}")
            return False

    # eval_and_upload.py - THAY THẾ hàm upload_stage_results()

    def upload_stage_results(
        self,
        stage:        str,
        algo_results: Dict,
        comparison:   Optional[Dict],
        stage_dir:    Path,
    ):
        """Upload tất cả JSON của 1 stage."""
        print(f"\n  📤 Uploading {stage.upper()} stage results...")
        
        # Upload từng algo
        for algo, result in algo_results.items():
            local_path = stage_dir / f"{algo}_{stage}.json"
            
            # ✅ FIX: Serialize AlgoResult → dict với FULL ARRAYS
            if hasattr(result, "__dataclass_fields__"):
                result_dict = asdict(result)
            else:
                result_dict = result
            
            # ✅ THÊM: Ensure arrays được lưu
            result_dict["full_data"] = {
                "ep_rewards":     result.rewards,      # ← List[float]
                "ep_coverages":   result.coverages,    # ← List[float]
                "ep_victim_rates": result.victim_rates, # ← List[float]
                "ep_successes":   result.successes,    # ← List[bool]
                "n_episodes":     len(result.rewards),
            }
            
            with open(local_path, "w") as f:
                json.dump(result_dict, f, indent=2, default=_json_safe)
            
            # Upload lên HF
            self.upload_file(
                str(local_path),
                f"eval_results/{stage}/{algo}_results.json",
                f"Eval {algo.upper()} on {stage} - Full episode data",
            )
        
        # Upload comparison
        if comparison:
            comp_path = stage_dir / f"comparison_{stage}.json"
            
            with open(comp_path, "w") as f:
                json.dump(comparison, f, indent=2, default=_json_safe)
            
            self.upload_file(
                str(comp_path),
                f"eval_results/{stage}/comparison.json",
                f"Statistical comparison {stage}",
            )

    # eval_and_upload.py - THAY THẾ phần tạo summary["stages"][stage][algo]

    def upload_summary(
        self,
        all_stage_results: Dict,
        out_dir: Path,
    ) -> Dict:
        """
        Tổng hợp EXTREME + TRANSFER → summary.json → upload.
        
        ✅ FIX: Lưu CẢ arrays và summary stats (giống MASAC trainer)
        """
        print(f"\n  📊 Creating summary (EXTREME + TRANSFER)...")
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "note":      "Eval on EXTREME (zero-shot) + TRANSFER (cross-domain). HARD stage used for training.",
            "stages":    {},
            "rankings":  {},
            "cross_stage_analysis": {},
        }

        # Extract metrics từng stage
        for stage, algo_results in all_stage_results.items():
            summary["stages"][stage] = {}
            
            for algo, r in algo_results.items():
                if hasattr(r, "reward_mean"):
                    # ✅ FIX: Lưu VỪA summary VỪA full arrays
                    summary["stages"][stage][algo] = {
                        # Summary stats (cho quick view)
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
                        
                        # ✅ THÊM: Full episode arrays (giống MASAC trainer)
                        "full_data": {
                            "ep_rewards":      [float(x) for x in r.rewards],
                            "ep_coverages":    [float(x) for x in r.coverages],
                            "ep_victim_rates": [float(x) for x in r.victim_rates],
                            "ep_successes":    [bool(x) for x in r.successes],
                        }
                    }


            # Rankings per stage
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

        # ✅ Cross-stage generalization (EXTREME → TRANSFER)
        if "extreme" in all_stage_results and "transfer" in all_stage_results:
            ext_r   = all_stage_results["extreme"]
            trans_r = all_stage_results["transfer"]
            
            summary["cross_stage_analysis"]["extreme_to_transfer"] = {}
            
            for algo in set(ext_r) & set(trans_r):
                e_cov = float(getattr(ext_r[algo],   "coverage_mean", 0))
                t_cov = float(getattr(trans_r[algo], "coverage_mean", 0))
                e_rew = float(getattr(ext_r[algo],   "reward_mean",   0))
                t_rew = float(getattr(trans_r[algo], "reward_mean",   0))
                
                summary["cross_stage_analysis"]["extreme_to_transfer"][algo] = {
                    "coverage_diff":     round(t_cov - e_cov, 4),
                    "reward_diff":       round(t_rew - e_rew, 2),
                    "coverage_transfer": round(t_cov / max(e_cov, 1e-8) * 100, 1),
                    "interpretation":    (
                        "Better on TRANSFER" if t_cov > e_cov
                        else "Better on EXTREME"
                    ),
                }

        # Print summary
        self._print_summary(summary)

        # Save + upload
        summary_path = out_dir / "summary.json"
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_safe)
        
        self.upload_file(
            str(summary_path),
            "eval_results/summary.json",
            "Evaluation summary (EXTREME + TRANSFER)",
        )

        return summary

    @staticmethod
    def _print_summary(summary: Dict):
        """In bảng tổng hợp kết quả."""
        print(f"\n{'═'*70}")
        print(f"  📊 EVALUATION SUMMARY (EXTREME + TRANSFER)")
        print(f"{'═'*70}")
        
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
        
        # Cross-stage analysis
        cross = summary.get("cross_stage_analysis", {}).get("extreme_to_transfer", {})
        if cross:
            print(f"\n  Cross-Stage Analysis (EXTREME → TRANSFER):")
            print(f"  {'-'*60}")
            for algo, data in cross.items():
                print(
                    f"    {algo.upper():<8}: "
                    f"cov_transfer={data['coverage_transfer']:.1f}%  "
                    f"({data['interpretation']})"
                )
        
        print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def download_checkpoints_from_hf(
    hf_token:  str,
    hf_repo:   str,
    local_dir: str,
    algos:     List[str] = ["mappo", "masac", "matd3"],
) -> Dict[str, str]:
    """Download checkpoint từ HuggingFace về local."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("❌ pip install huggingface-hub")
        return {}

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  🔍 Listing files trong HF repo: {hf_repo}")
    
    try:
        all_files = list(list_repo_files(
            hf_repo,
            token     = hf_token,
            repo_type = "dataset",
        ))
        
        pt_files = [f for f in all_files if f.endswith(".pt")]
        print(f"  📋 Tìm thấy {len(pt_files)} file .pt:")
        for f in pt_files[:10]:
            print(f"       {f}")
        if len(pt_files) > 10:
            print(f"       ... và {len(pt_files)-10} files nữa")
            
    except Exception as e:
        print(f"  ❌ Không list được files: {e}")
        all_files = []

    downloaded = {}

    for algo in algos:
        print(f"\n  📥 Tìm checkpoint cho {algo.upper()}...")

        algo_pt_files = [
            f for f in all_files
            if algo in f.lower() and f.endswith(".pt")
        ]

        if not algo_pt_files:
            print(f"  ❌ {algo}: Không tìm thấy .pt file nào")
            continue

        # Ưu tiên checkpoint_final
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
            print(f"  ❌ {algo}: Download failed: {e}")

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    hf_token:    str,
    hf_repo:     str,
    checkpoints: Optional[Dict[str, str]] = None,
    stages:      List[str]                = ["extreme", "transfer"],  # ← DEFAULT chỉ 2 stages
    n_episodes:  int                      = 100,
    base_seed:   int                      = 9999,
    device:      str                      = "cpu",
    n_envs:      int                      = 1,
    n_uav:       int                      = 4,
    output_dir:  str                      = "results/eval",
) -> Dict:
    """
    Eval 3 algo trên EXTREME + TRANSFER → lưu JSON → upload HF.
    
    EXTREME:  Zero-shot eval (khó hơn HARD)
    TRANSFER: Cross-domain eval
    
    NOTE: HARD stage đã dùng để train, không eval lại.
    """
    from evaluate import build_eval_config, evaluate_algo, compare_algos

    print(f"\n{'═'*70}")
    print(f"  🚁 SAR UAV SWARM — EVALUATION (EXTREME + TRANSFER)")
    print(f"{'═'*70}")
    print(f"  Stages     : {stages}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Parallel   : {n_envs} envs")
    print(f"  Device     : {device}")
    print(f"  HF Repo    : {hf_repo}")
    print(f"{'═'*70}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uploader = EvalHFUploader(hf_token, hf_repo)

    # Download nếu chưa có checkpoint
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
        print(f"\n{'─'*70}")
        print(f"  📍 STAGE: {stage.upper()}")
        print(f"{'─'*70}")

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

        # Upload stage results
        uploader.upload_stage_results(stage, algo_results, comparison, stage_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = uploader.upload_summary(all_stage_results, out_dir)

    # ── Full dump ─────────────────────────────────────────────────────────────
    full = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "note":        "Eval on EXTREME (zero-shot) + TRANSFER (cross-domain). HARD used for training.",
        "config":      {
            "stages":     stages,
            "n_episodes": n_episodes,
            "n_envs":     n_envs,
            "device":     device
        },
        "checkpoints": {a: str(p) for a, p in checkpoints.items()},
        "summary":     summary,
    }
    
    full_path = out_dir / "full_eval_results.json"
    with open(full_path, "w") as f:
        json.dump(full, f, indent=2, default=_json_safe)
    
    uploader.upload_file(
        str(full_path),
        "eval_results/full_eval_results.json",
        "Full evaluation results (EXTREME + TRANSFER)",
    )

    print(f"\n{'═'*70}")
    print(f"  ✅ DONE!")
    print(f"  📁 Local : {out_dir}")
    print(f"  ☁️  HF    : https://huggingface.co/datasets/{hf_repo}/tree/main/eval_results")
    print(f"{'═'*70}\n")

    return {"algo_results": all_stage_results, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Eval MAPPO/MASAC/MATD3 trên EXTREME + TRANSFER và upload HF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # HuggingFace
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
        help     = "HF repo ID, vd: username/sar-uav-results",
    )
    p.add_argument(
        "--hf-ckpt-repo",
        type    = str,
        default = None,
        help    = "HF repo chứa checkpoint (nếu khác --hf-repo)",
    )

    # Checkpoint local
    p.add_argument("--mappo", type=str, default=None)
    p.add_argument("--masac", type=str, default=None)
    p.add_argument("--matd3", type=str, default=None)

    # Eval params
    p.add_argument(
        "--stages",
        nargs   = "+",
        default = ["extreme", "transfer"],  # ← MẶC ĐỊNH CHỈ 2 STAGES
        choices = ["hard", "extreme", "transfer"],  # Vẫn cho phép hard nếu cần
        help    = "Stages để eval (default: extreme transfer)",
    )
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--n-envs",     type=int, default=1)
    p.add_argument("--seed",       type=int, default=9999)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--n-uav",      type=int, default=4)

    # Output
    p.add_argument("--output-dir", type=str, default="results/eval")

    return p.parse_args()


def main():
    args = parse_args()

    # Lấy HF token
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

    ckpt_repo = args.hf_ckpt_repo or args.hf_repo

    # Build checkpoints
    checkpoints: Dict[str, str] = {}

    for algo, path in [
        ("mappo", args.mappo),
        ("masac", args.masac),
        ("matd3", args.matd3),
    ]:
        if path and Path(path).exists():
            checkpoints[algo] = str(Path(path))
            print(f"  ✅ {algo.upper()}: local checkpoint")

    # Download missing checkpoints
    missing = [a for a in ["mappo", "masac", "matd3"]
               if a not in checkpoints]
    if missing:
        print(f"\n  📥 Download {missing} từ: {ckpt_repo}")
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

    # Run
    run_full_evaluation(
        hf_token    = hf_token,
        hf_repo     = args.hf_repo,
        checkpoints = checkpoints,
        stages      = args.stages,  # Chỉ extreme + transfer
        n_episodes  = args.n_episodes,
        base_seed   = args.seed,
        device      = args.device,
        n_envs      = args.n_envs,
        n_uav       = args.n_uav,
        output_dir  = args.output_dir,
    )


if __name__ == "__main__":
    main()