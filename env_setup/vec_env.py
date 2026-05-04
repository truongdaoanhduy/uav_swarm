"""
env_setup/vec_env.py
Vectorized environment cho MAPPO training.
Fixed bugs:
    1. Config serialization qua pickle
    2. Multiprocessing start method
    3. Obs/info rỗng khi done
"""

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict
import copy


def env_worker(pipe, config_dict, seed):
    """
    Worker process: chạy 1 env, nhận lệnh qua pipe.
    
    Nhận config_dict thay vì config object để tránh pickle issues.
    """
    try:
        from env_setup.sar_pettingzoo_env import SARPettingZooEnv
        from config import AppConfig
        
        # Reconstruct config từ dict
        config = config_dict  # AppConfig đã picklable nếu dùng dataclass
        
        n_agents = config.env.n_uav
        obs_dim = config.obs.actor_dim
        global_obs_dim = config.obs.critic_dim
        
        env = SARPettingZooEnv(config, render_mode=None)
        
        # Cache last valid state
        last_obs_array = np.zeros((n_agents, obs_dim), dtype=np.float32)
        last_global_obs = np.zeros(global_obs_dim, dtype=np.float32)
        last_info = {
            'uav_0': {
                'coverage_rate': 0.0,
                'victims_found': 0,
                'victims_total': 1,  # Tránh div by zero
                'global_obs': np.zeros(global_obs_dim, dtype=np.float32)
            }
        }
        
        # Initial reset
        obs, info = env.reset(seed=seed)
        if obs:
            agent_ids = sorted(obs.keys())
            last_obs_array = np.stack([obs[aid] for aid in agent_ids], axis=0)
        if info and 'uav_0' in info:
            last_global_obs = info['uav_0']['global_obs'].copy()
            last_info = info
        
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "reset":
                obs, info = env.reset(seed=seed)
                if obs:
                    agent_ids = sorted(obs.keys())
                    last_obs_array = np.stack(
                        [obs[aid] for aid in agent_ids], axis=0
                    ).astype(np.float32)
                if info and 'uav_0' in info:
                    last_global_obs = info['uav_0']['global_obs'].copy()
                    last_info = info
                
                pipe.send((
                    last_obs_array.copy(),
                    last_global_obs.copy(),
                    last_info
                ))
                
            elif cmd == "step":
                actions = data  # [n_agents, 3]
                actions_dict = {
                    f"uav_{i}": actions[i] for i in range(n_agents)
                }
                
                obs, rewards, terms, truncs, info = env.step(actions_dict)
                done = any(terms.values()) or any(truncs.values())
                
                # Update cache nếu obs hợp lệ
                if obs and len(obs) > 0:
                    # Lọc agents còn trong obs
                    valid_agents = sorted(obs.keys())
                    if len(valid_agents) > 0:
                        obs_arrays = []
                        for i in range(n_agents):
                            aid = f"uav_{i}"
                            if aid in obs:
                                obs_arrays.append(obs[aid])
                            else:
                                # Agent đã done → dùng zero obs
                                obs_arrays.append(
                                    np.zeros(obs_dim, dtype=np.float32)
                                )
                        last_obs_array = np.stack(
                            obs_arrays, axis=0
                        ).astype(np.float32)
                
                if info and 'uav_0' in info:
                    last_global_obs = info['uav_0']['global_obs'].copy()
                    last_info = info
                
                # Extract rewards
                rewards_array = np.zeros(n_agents, dtype=np.float32)
                if rewards:
                    for i in range(n_agents):
                        aid = f"uav_{i}"
                        if aid in rewards:
                            rewards_array[i] = rewards[aid]
                
                # Gửi kết quả TRƯỚC khi reset
                pipe.send((
                    last_obs_array.copy(),
                    last_global_obs.copy(),
                    rewards_array.copy(),
                    done,
                    last_info
                ))
                
                # Auto reset sau khi gửi
                if done:
                    obs_new, info_new = env.reset(seed=seed)
                    if obs_new:
                        agent_ids = sorted(obs_new.keys())
                        last_obs_array = np.stack(
                            [obs_new[aid] for aid in agent_ids], axis=0
                        ).astype(np.float32)
                    if info_new and 'uav_0' in info_new:
                        last_global_obs = info_new['uav_0']['global_obs'].copy()
                        last_info = info_new
                    
            elif cmd == "close":
                break
                
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as e:
        import traceback
        print(f"\n[Worker {seed} ERROR] {e}")
        traceback.print_exc()
        # Gửi signal lỗi nếu có thể
        try:
            pipe.send(None)  # Sentinel value
        except:
            pass
    finally:
        try:
            env.close()
        except:
            pass
        try:
            pipe.close()
        except:
            pass


class VectorizedEnv:
    """
    Chạy N environments song song bằng multiprocessing.
    
    Fixed:
        - Buffer overflow: mỗi step() → đúng 1 transition per env
        - Start method: spawn thay vì fork (CUDA safe)
        - Error handling: worker crash detection
    """
    
    def __init__(self, config, n_envs: int = 8, start_seed: int = 0):
        self.n_envs = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim
        self.action_dim = 3
        self.config = config
        self.start_seed = start_seed
        
        # Dùng spawn để tránh CUDA fork issues
        ctx = mp.get_context("spawn")
        
        self.pipes = []
        self.processes = []
        
        print(f"  🔧 Creating {n_envs} parallel environments (spawn method)...")
        
        for i in range(n_envs):
            parent_pipe, child_pipe = ctx.Pipe()
            p = ctx.Process(
                target=env_worker,
                args=(child_pipe, config, start_seed + i),
                daemon=True
            )
            p.start()
            child_pipe.close()  # Đóng child pipe ở parent
            self.pipes.append(parent_pipe)
            self.processes.append(p)
        
        # Verify workers alive
        import time
        time.sleep(0.5)
        alive = sum(1 for p in self.processes if p.is_alive())
        print(f"  ✅ {alive}/{n_envs} environment workers ready!")
        
        if alive < n_envs:
            raise RuntimeError(f"Only {alive}/{n_envs} workers started successfully!")
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset tất cả envs.
        
        Returns:
            obs_batch:        [n_envs, n_agents, obs_dim]
            global_obs_batch: [n_envs, global_obs_dim]
        """
        for pipe in self.pipes:
            pipe.send(("reset", None))
        
        obs_list = []
        global_obs_list = []
        
        for i, pipe in enumerate(self.pipes):
            result = pipe.recv()
            if result is None:
                raise RuntimeError(f"Worker {i} crashed during reset!")
            obs, global_obs, info = result
            obs_list.append(obs)
            global_obs_list.append(global_obs)
        
        return (
            np.stack(obs_list, axis=0),        # [n_envs, n_agents, obs_dim]
            np.stack(global_obs_list, axis=0)  # [n_envs, global_obs_dim]
        )
    
    def step(
        self,
        actions_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[Dict]]:
        """
        Step tất cả envs song song.
        
        Args:
            actions_batch: [n_envs, n_agents, action_dim]
        
        Returns:
            obs_batch:        [n_envs, n_agents, obs_dim]
            global_obs_batch: [n_envs, global_obs_dim]
            rewards_batch:    [n_envs, n_agents]
            dones:            List[bool] len=n_envs
            infos:            List[Dict] len=n_envs
        """
        # Gửi async
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions_batch[i]))
        
        # Thu kết quả
        obs_list, global_obs_list, rewards_list = [], [], []
        dones, infos = [], []
        
        for i, pipe in enumerate(self.pipes):
            result = pipe.recv()
            if result is None:
                raise RuntimeError(f"Worker {i} crashed during step!")
            obs, global_obs, rewards, done, info = result
            obs_list.append(obs)
            global_obs_list.append(global_obs)
            rewards_list.append(rewards)
            dones.append(done)
            infos.append(info)
        
        return (
            np.stack(obs_list, axis=0),        # [n_envs, n_agents, obs_dim]
            np.stack(global_obs_list, axis=0), # [n_envs, global_obs_dim]
            np.stack(rewards_list, axis=0),    # [n_envs, n_agents]
            dones,
            infos
        )
    
    def close(self):
        """Đóng tất cả processes an toàn."""
        for i, pipe in enumerate(self.pipes):
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        
        for p in self.processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        
        print("  ✅ All env workers closed.")
    
    def __del__(self):
        """Cleanup khi object bị destroy."""
        try:
            self.close()
        except Exception:
            pass