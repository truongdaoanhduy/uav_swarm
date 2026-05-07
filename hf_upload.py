"""
HuggingFace Upload/Download Utility
Token đọc từ environment variable hoặc truyền vào
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# UPLOADER
# ══════════════════════════════════════════════════════════════════════════════

class HFUploader:
    def __init__(self, token: str = None, repo_id: str = None):
        """
        Args:
            token: HF token (nếu None → đọc từ HF_TOKEN env var)
            repo_id: Repo ID (nếu None → đọc từ HF_REPO env var)
        """
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "❌ HuggingFace token not found!\n"
                "   Set via:\n"
                "   - Environment variable: export HF_TOKEN=hf_xxx\n"
                "   - Kaggle Secrets: HF_TOKEN\n"
                "   - CLI argument: --hf-token hf_xxx"
            )

        self.repo_id = repo_id or os.getenv("HF_REPO", "duy95/sar-uav-results")
        self._ensure_repo()

    def _get_api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _ensure_repo(self):
        api = self._get_api()
        try:
            api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(
                repo_id=self.repo_id, repo_type="dataset",
                private=False, exist_ok=True,
            )

    def upload_checkpoint(
        self,
        checkpoint_path: Path,
        run_name:        str,
        episode:         int,
        metrics:         Dict = None,
    ):
        """Upload 1 checkpoint (periodic)."""
        api      = self._get_api()
        cp_path  = Path(checkpoint_path)
        hf_path  = f"{run_name}/{cp_path.name}"

        print(f"📤 Uploading checkpoint → HF: {hf_path}")
        api.upload_file(
            path_or_fileobj = str(cp_path),
            path_in_repo    = hf_path,
            repo_id         = self.repo_id,
            repo_type       = "dataset",
        )

        if metrics:
            self._upload_metrics(run_name, metrics, episode)

        print(f"   ✅ https://huggingface.co/datasets/{self.repo_id}/blob/main/{hf_path}")

    def upload_final(
        self,
        run_name:        str,
        checkpoint_path: Path,
        metrics:         Dict,
        plot_path:       Path = None,
    ):
        """Upload final checkpoint + full metrics + plot."""
        api     = self._get_api()
        cp_path = Path(checkpoint_path)

        # 1. Checkpoint
        hf_cp = f"{run_name}/checkpoint_final.pt"
        print(f"📤 Uploading final checkpoint → {hf_cp}")
        api.upload_file(
            path_or_fileobj = str(cp_path),
            path_in_repo    = hf_cp,
            repo_id         = self.repo_id,
            repo_type       = "dataset",
        )

        # 2. Metrics JSON (full curves)
        self._upload_metrics(run_name, metrics, episode=metrics.get("total_episodes", 0))

        # 3. Plot PNG
        if plot_path and Path(plot_path).exists():
            hf_plot = f"{run_name}/training_curves.png"
            print(f"📤 Uploading plot → {hf_plot}")
            api.upload_file(
                path_or_fileobj = str(plot_path),
                path_in_repo    = hf_plot,
                repo_id         = self.repo_id,
                repo_type       = "dataset",
            )

        print(f"\n🎉 Final upload complete!")
        print(f"   https://huggingface.co/datasets/{self.repo_id}/tree/main/{run_name}")

    def _upload_metrics(self, run_name: str, metrics: Dict, episode: int):
        """Upload metrics.json."""
        import tempfile
        api      = self._get_api()
        hf_path  = f"{run_name}/metrics.json"

        # Serialize (convert numpy → python native)
        safe_metrics = _serialize(metrics)
        safe_metrics["uploaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_metrics["episode"]     = episode

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(safe_metrics, f, indent=2)
            tmp = f.name

        api.upload_file(
            path_or_fileobj = tmp,
            path_in_repo    = hf_path,
            repo_id         = self.repo_id,
            repo_type       = "dataset",
        )
        os.unlink(tmp)
        print(f"   📊 Metrics uploaded → {hf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

class HFDownloader:
    """Download checkpoints + metrics từ HuggingFace."""

    def __init__(
        self,
        token:   str = None,
        repo_id: str = None,
    ):
        self.token   = token or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or os.getenv("HF_REPO", "duy95/sar-uav-results")

        if not self.token:
            raise ValueError("HF_TOKEN not found. Set environment variable or pass token.")

    def list_runs(self) -> List[str]:
        """Liệt kê tất cả run_names có trên HF."""
        from huggingface_hub import HfApi
        api   = HfApi(token=self.token)
        files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")

        runs = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2:
                runs.add(parts[0])
        return sorted(runs)

    def download_metrics(self, run_name: str, local_dir: str = ".") -> Dict:
        """Download metrics.json cho 1 run."""
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id  = self.repo_id,
            filename = f"{run_name}/metrics.json",
            repo_type= "dataset",
            token    = self.token,
            local_dir= local_dir,
        )
        with open(local) as f:
            return json.load(f)

    def download_checkpoint(
        self,
        run_name: str,
        filename: str = "checkpoint_final.pt",
        local_dir: str = ".",
    ) -> str:
        """Download checkpoint. Trả về local path."""
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id  = self.repo_id,
            filename = f"{run_name}/{filename}",
            repo_type= "dataset",
            token    = self.token,
            local_dir= local_dir,
        )
        print(f"✅ Downloaded: {local}")
        return local

    def download_all_metrics(self, local_dir: str = ".") -> Dict[str, Dict]:
        """Download metrics của tất cả runs. Trả về {run_name: metrics}."""
        runs    = self.list_runs()
        results = {}
        print(f"📥 Found {len(runs)} runs: {runs}")

        for run in runs:
            try:
                metrics        = self.download_metrics(run, local_dir)
                results[run]   = metrics
                print(f"   ✅ {run}")
            except Exception as e:
                print(f"   ⚠️  {run}: {e}")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _serialize(obj):
    """Convert numpy types → Python native cho JSON."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj