"""
env_setup/vec_env.py
FIXED: Seed progression cho mỗi episode
"""

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict


def env_worker(pipe, config_dict, seed):
    """
    Worker process với seed progression.
    
    FIXED: Mỗi episode dùng seed khác nhau
    """
    try:
        from env_setup.sar_pettingzoo_env import SARPettingZooEnv
        from config import AppConfig
        
        config = config_dict
        n_agents = config.env.n_uav
        obs_dim = config.obs.actor_dim
        global_obs_dim = config.obs.critic_dim
        
        env = SARPettingZooEnv(config, render_mode=None)
        
        # Cache
        last_obs_array = np.zeros((n_agents, obs_dim), dtype=np.float32)
        last_global_obs = np.zeros(global_obs_dim, dtype=np.float32)
        last_info = {
            'uav_0': {
                'coverage_rate':  0.0,
                'victims_found':  0,
                'victims_total':  1,
                'global_obs':     np.zeros(global_obs_dim, dtype=np.float32),
                'done_reason':    None,
                'success':        False,
                'battery_stats':  {'mean': 100.0, 'min': 100.0, 'max': 100.0, 'std': 0.0},
                'n_active':       config.env.n_uav,
                'n_returning':    0,
                'n_charging':     0,
                'n_deploying':    0,
                'n_disabled':     0,
                'n_total_uavs':   config.env.n_uav,
                'n_accounted':    config.env.n_uav,
                'step':           0,
                'episode':        {
                    'total_landings':    0,
                    'total_charge_time': 0,
                    'landings_per_uav':  {},
                },
            }
        }
        
        # ✅ FIX: Episode counter + RNG cho seed progression
        episode_count = 0
        rng = np.random.default_rng(seed)  # ← RNG riêng cho worker
        
        # Initial reset với seed gốc
        current_seed = seed
        obs, info = env.reset(seed=current_seed)
        episode_count += 1
        
        if obs:
            agent_ids = sorted(obs.keys())
            last_obs_array = np.stack([obs[aid] for aid in agent_ids], axis=0)
        if info and 'uav_0' in info:
            last_global_obs = info['uav_0']['global_obs'].copy()
            last_info = info
        
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "reset":
                # ✅ FIX: Generate seed mới cho mỗi episode
                current_seed = int(rng.integers(0, 2**31))
                obs, info = env.reset(seed=current_seed)
                episode_count += 1
                
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
                actions = data  # [n_agents, 4]
                actions_dict = {
                    f"uav_{i}": actions[i] for i in range(n_agents)
                }
                
                obs, rewards, terms, truncs, info = env.step(actions_dict)
                done = any(terms.values()) or any(truncs.values())

                # Update cache nếu obs hợp lệ
                if obs and len(obs) > 0:
                    obs_arrays = []
                    for i in range(n_agents):
                        aid = f"uav_{i}"
                        if aid in obs:
                            obs_arrays.append(obs[aid])
                        else:
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
                
                # Send kết quả TRƯỚC khi reset
                pipe.send((
                    last_obs_array.copy(),
                    last_global_obs.copy(),
                    rewards_array.copy(),
                    done,
                    last_info
                ))
                
                # ✅ Reset sau done — base_env KHÔNG tự reset nữa nên chỉ reset 1 lần
                if done:
                    current_seed = int(rng.integers(0, 2**31))
                    obs_new, info_new = env.reset(seed=current_seed)
                    episode_count += 1
                    
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
        print(f"\n[Worker ERROR] {e}")
        traceback.print_exc()
        try:
            pipe.send(None)
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

from numpy.random import SeedSequence

class VectorizedEnv:
    """Vectorized environment với seed progression fix."""
    
    def __init__(self, config, n_envs: int = 8, start_seed: int = 0):
        self.n_envs = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim
        self.action_dim = 4
        self.config = config
        self.start_seed = start_seed
        
        ctx = mp.get_context("spawn")
        
        self.pipes = []
        self.processes = []
        
        print(f"  🔧 Creating {n_envs} parallel environments (spawn method)...")
        
        ss = SeedSequence(start_seed)
        worker_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_envs)]

        for i in range(n_envs):
            parent_pipe, child_pipe = ctx.Pipe()
            p = ctx.Process(
                target=env_worker,
                args=(child_pipe, config, worker_seeds[i]),  # ← Dùng worker_seeds
                daemon=True
            )

            p.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(p)
        
        import time
        time.sleep(0.5)
        alive = sum(1 for p in self.processes if p.is_alive())
        print(f"  ✅ {alive}/{n_envs} environment workers ready!")
        
        if alive < n_envs:
            raise RuntimeError(f"Only {alive}/{n_envs} workers started!")
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
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
            np.stack(obs_list, axis=0),
            np.stack(global_obs_list, axis=0)
        )
    
    def step(
        self,
        actions_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[Dict]]:
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions_batch[i]))
        
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
            np.stack(obs_list, axis=0),
            np.stack(global_obs_list, axis=0),
            np.stack(rewards_list, axis=0),
            dones,
            infos
        )
    
    def close(self):
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
        try:
            self.close()
        except Exception:
            pass