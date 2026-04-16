"""
UAV Crowdsensing Environment (Section III of the paper)

Simulates:
- Time-slotted system with T timeslots
- UAVs with limited energy, moving at configurable speed
- Surveillance PoIs (fixed, periodic data generation)
- Emergency PoIs (unpredictable, time-critical)
- AoI tracking and valid task handling index computation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import copy


@dataclass
class SimConfig:
    """Simulation configuration matching Table II."""
    # World
    world_size: float = 6000.0        # 6km x 6km area
    timeslot_duration: float = 20.0   # seconds per timeslot (τ)
    max_timeslots: int = 120          # T

    # UAVs
    num_uavs: int = 4                 # U
    max_energy: float = 500000.0      # Emax (Joules)
    max_speed: float = 15.0           # vmax (m/s)
    energy_coeff: float = 0.005       # αε energy consumption per timeslot
    uav_sensing_range: float = 500.0  # sensing range for AoI heatmap

    # Surveillance PoIs
    num_surv_pois: int = 300
    surv_aoi_threshold: int = 35      # AoIsurv_th (timeslots)
    data_size: float = 1.0            # D (normalized)

    # Emergency PoIs
    emer_aoi_threshold: int = 20      # AoIemer_th (timeslots)
    emer_area_size: float = 200.0     # l_emer (meters)
    emer_interval: int = 6            # Δ (timeslots between emergencies)
    max_image_blur: float = 5.0       # δ_max (pixels)
    alpha_img: float = 0.1            # α_img (blur coefficient)
    camera_radius: float = 50.0       # ρ (meters)

    # Communication (simplified)
    bandwidth: float = 1e6            # B (Hz)
    tx_power: float = 23.0            # Wtx (dBm)
    noise_power: float = -90.0        # Wn (dBm)
    min_snr: float = 5.0             # Wmin (dB)
    freq_factor: float = 0.1         # αf

    # LoS parameters (urban)
    alpha1: float = 9.61
    alpha2: float = 0.16
    los_loss: float = 1.0             # α_LoS (dB)
    nlos_loss: float = 20.0           # α_NLoS (dB)

    # Algorithm
    queue_length: int = 3             # l_que
    omega: float = 0.7                # self-balancing reward weight
    grid_resolution: int = 16         # AoI heatmap grid size


@dataclass
class PoI:
    """Point of Interest."""
    poi_id: int
    x: float
    y: float
    is_emergency: bool
    aoi_threshold: int
    aoi: int = 1                      # current AoI
    active: bool = True               # emergency PoIs become inactive after handling
    handled: bool = False
    arrival_time: int = 0             # timeslot when PoI appeared


@dataclass
class UAVState:
    """State of a single UAV."""
    uav_id: int
    x: float
    y: float
    energy: float
    speed: float = 0.0


class UAVCrowdsensingEnv:
    """
    Multi-task-oriented UAV crowdsensing environment.

    Implements the system model from Section III:
    - Surveillance PoIs at fixed locations (generate-at-will mode)
    - Emergency PoIs appearing at random locations/interval Δ
    - AoI tracking (Eq. 5)
    - Valid handling ratio (Eq. 6)
    - Valid task handling index (Eq. 7)
    - Energy consumption model
    """

    def __init__(self, config: Optional[SimConfig] = None, seed: int = 42):
        self.cfg = config or SimConfig()
        self.rng = np.random.RandomState(seed)

        # PoIs
        self.surv_pois: List[PoI] = []
        self.emer_pois: List[PoI] = []
        self.all_pois: List[PoI] = []

        # UAVs
        self.uavs: List[UAVState] = []

        # Time
        self.current_timeslot: int = 0
        self.next_poi_id: int = 0
        self.next_emer_time: int = 0

        # For reward computation
        self._prev_surv_handled = 0
        self._prev_emer_handled = 0
        self._emer_just_assigned: List[int] = []

        # History
        self.aoi_history: Dict[int, List[int]] = {}  # poi_id -> list of AoI values

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset the environment."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_timeslot = 0
        self.next_poi_id = 0
        self.next_emer_time = 0
        self._prev_surv_handled = 0
        self._prev_emer_handled = 0
        self._emer_just_assigned = []
        self.aoi_history = {}

        # Initialize surveillance PoIs
        self.surv_pois = []
        for _ in range(self.cfg.num_surv_pois):
            x = self.rng.uniform(100, self.cfg.world_size - 100)
            y = self.rng.uniform(100, self.cfg.world_size - 100)
            poi = PoI(
                poi_id=self.next_poi_id,
                x=x, y=y,
                is_emergency=False,
                aoi_threshold=self.cfg.surv_aoi_threshold,
                aoi=1,
                arrival_time=0
            )
            self.next_poi_id += 1
            self.surv_pois.append(poi)
            self.aoi_history[poi.poi_id] = [1]

        self.emer_pois = []
        self.all_pois = list(self.surv_pois)

        # Initialize UAVs at random positions
        self.uavs = []
        for i in range(self.cfg.num_uavs):
            uav = UAVState(
                uav_id=i,
                x=self.rng.uniform(500, self.cfg.world_size - 500),
                y=self.rng.uniform(500, self.cfg.world_size - 500),
                energy=self.cfg.max_energy
            )
            self.uavs.append(uav)

        # Generate first emergency
        self._generate_emergency()
        self.next_emer_time = self.cfg.emer_interval

        return self._get_obs()

    def _generate_emergency(self):
        """Generate a new emergency PoI at a random location."""
        x = self.rng.uniform(100, self.cfg.world_size - 100)
        y = self.rng.uniform(100, self.cfg.world_size - 100)
        poi = PoI(
            poi_id=self.next_poi_id,
            x=x, y=y,
            is_emergency=True,
            aoi_threshold=self.cfg.emer_aoi_threshold,
            aoi=1,
            active=True,
            arrival_time=self.current_timeslot
        )
        self.next_poi_id += 1
        self.emer_pois.append(poi)
        self.all_pois.append(poi)
        self.aoi_history[poi.poi_id] = [1]
        self._emer_just_assigned.append(poi.poi_id)
        return poi

    def step(self, actions: List[Tuple[float, float, float]]) -> Tuple[dict, List[float], bool, dict]:
        """
        Execute one timeslot.

        Args:
            actions: List of (dx, dy, speed) for each UAV.
                     dx, dy in [-1, 1], speed in [0, 1] (normalized).

        Returns:
            obs, rewards, done, info
        """
        self.current_timeslot += 1
        rewards = []

        # 1. Move UAVs
        for i, uav in enumerate(self.uavs):
            if uav.energy <= 0:
                rewards.append(0.0)
                continue

            dx, dy, spd_norm = actions[i]
            speed = max(0.1, min(spd_norm, 1.0)) * self.cfg.max_speed

            # Normalize direction
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 0:
                dx /= mag
                dy /= mag

            # Move
            dist = speed * self.cfg.timeslot_duration
            uav.x = np.clip(uav.x + dx * dist, 0, self.cfg.world_size)
            uav.y = np.clip(uav.y + dy * dist, 0, self.cfg.world_size)
            uav.speed = speed

            # Energy consumption (Eq. in Section III)
            energy_cost = self.cfg.energy_coeff * self.cfg.timeslot_duration
            uav.energy = max(0, uav.energy - energy_cost)

        # 2. Handle surveillance PoIs (generate-at-will)
        surv_handled_this_step = 0
        for poi in self.surv_pois:
            best_uav = None
            best_snr = -np.inf
            for uav in self.uavs:
                if uav.energy <= 0:
                    continue
                dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                if dist < self.cfg.camera_radius:
                    # Simplified: nearby UAV handles it
                    snr = self.cfg.tx_power - self.cfg.noise_power - 20 * np.log10(max(dist, 1))
                    if snr > best_snr:
                        best_snr = snr
                        best_uav = uav
            if best_uav is not None:
                poi.handled = True
                poi.aoi = 1
                surv_handled_this_step += 1
            else:
                poi.aoi = min(poi.aoi + 1, self.cfg.surv_aoi_threshold + 10)

            if poi.poi_id in self.aoi_history:
                self.aoi_history[poi.poi_id].append(poi.aoi)

        # 3. Handle emergency PoIs
        emer_handled_this_step = 0
        for poi in self.emer_pois:
            if not poi.active:
                continue

            for uav in self.uavs:
                if uav.energy <= 0:
                    continue
                dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                if dist <= self.cfg.emer_area_size:
                    # Check image blur constraint (Section III-B)
                    if uav.speed <= self.cfg.max_image_blur / self.cfg.alpha_img:
                        # Handle emergency: zigzag trajectory simulation
                        handle_time = self.cfg.emer_area_size**2 / (2 * uav.speed * self.cfg.camera_radius)
                        if handle_time <= self.cfg.timeslot_duration:
                            poi.active = False
                            poi.handled = True
                            poi.aoi = 1
                            emer_handled_this_step += 1
                            break

            if poi.active:
                poi.aoi = min(poi.aoi + 1, self.cfg.emer_aoi_threshold + 10)

            if poi.poi_id in self.aoi_history:
                self.aoi_history[poi.poi_id].append(poi.aoi)

        # 4. Generate new emergencies
        if self.current_timeslot >= self.next_emer_time:
            self._generate_emergency()
            self.next_emer_time = self.current_timeslot + self.cfg.emer_interval

        # 5. Compute rewards (Section IV-B, Eq. 13)
        for i, uav in enumerate(self.uavs):
            if uav.energy <= 0:
                rewards.append(0.0)
                continue

            # Emergency reward
            r_emer = 0.0
            for poi in self.emer_pois:
                if not poi.active and poi.arrival_time == self.current_timeslot - 1:
                    r_emer += 1.0
                elif poi.active and poi.aoi > poi.aoi_threshold and poi.arrival_time == self.current_timeslot - 1:
                    r_emer -= 1.0

            # Surveillance reward (AoI difference normalized)
            r_surv = 0.0
            count = 0
            for poi in self.surv_pois:
                dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                if dist < self.cfg.camera_radius * 2:
                    hist = self.aoi_history.get(poi.poi_id, [])
                    if len(hist) >= 2:
                        aoi_diff = (hist[-2] - hist[-1]) / self.cfg.max_timeslots
                        r_surv += aoi_diff
                        count += 1
            if count > 0:
                r_surv /= count

            # Energy penalty (Section IV-B)
            r_eps = -0.01 if uav.energy < self.cfg.max_energy * 0.1 else 0.0

            r_low = r_emer + r_surv - r_eps
            rewards.append(r_low)

        self._prev_surv_handled = surv_handled_this_step
        self._prev_emer_handled = emer_handled_this_step

        # Check termination
        done = self.current_timeslot >= self.cfg.max_timeslots

        obs = self._get_obs()
        info = self._get_info()
        return obs, rewards, done, info

    def _get_obs(self) -> dict:
        """Build observation dict for all UAVs."""
        obs = {}
        for uav in self.uavs:
            # UAV local state
            uav_obs = {
                'uav_id': uav.uav_id,
                'position': np.array([uav.x / self.cfg.world_size, uav.y / self.cfg.world_size]),
                'energy': np.array([uav.energy / self.cfg.max_energy]),
                'speed': np.array([uav.speed / self.cfg.max_speed]),
            }

            # Local AoI heatmap (grid of surveillance PoI AoI values)
            grid = self._build_aoi_heatmap(uav)
            uav_obs['aoi_heatmap'] = grid

            # Relative positions to other UAVs
            rel_positions = []
            for other in self.uavs:
                if other.uav_id != uav.uav_id:
                    rx = (other.x - uav.x) / self.cfg.world_size
                    ry = (other.y - uav.y) / self.cfg.world_size
                    rel_positions.extend([rx, ry])
            uav_obs['rel_uav_positions'] = np.array(rel_positions)

            # Emergency PoI features (for goal assignment)
            emer_features = []
            for poi in self.emer_pois:
                if poi.active:
                    dx = (poi.x - uav.x) / self.cfg.world_size
                    dy = (poi.y - uav.y) / self.cfg.world_size
                    remaining = max(0, poi.aoi_threshold - poi.aoi) / poi.aoi_threshold
                    emer_features.extend([dx, dy, remaining])
            if not emer_features:
                emer_features = [0, 0, 0]
            uav_obs['emer_features'] = np.array(emer_features)

            obs[uav.uav_id] = uav_obs

        # Global state for high-level allocator
        global_state = {
            'timeslot': np.array([self.current_timeslot / self.cfg.max_timeslots]),
            'uav_positions': np.array([
                [u.x / self.cfg.world_size, u.y / self.cfg.world_size] for u in self.uavs
            ]).flatten(),
            'uav_energies': np.array([u.energy / self.cfg.max_energy for u in self.uavs]),
        }

        # Pending emergency PoIs
        pending_emer = [p for p in self.emer_pois if p.active]
        if pending_emer:
            global_state['emer_positions'] = np.array([
                [p.x / self.cfg.world_size, p.y / self.cfg.world_size, p.aoi / p.aoi_threshold]
                for p in pending_emer
            ]).flatten()
        else:
            global_state['emer_positions'] = np.array([0.0])

        obs['global'] = global_state
        obs['pending_emer_pois'] = pending_emer
        obs['timeslot'] = self.current_timeslot
        return obs

    def _build_aoi_heatmap(self, uav: UAVState) -> np.ndarray:
        """Build local AoI heatmap for a UAV (Section III-D observation)."""
        res = self.cfg.grid_resolution
        grid = np.zeros((res, res), dtype=np.float32)
        cell_size = self.cfg.world_size / res

        # Only consider PoIs within sensing range
        for poi in self.surv_pois:
            dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
            if dist <= self.cfg.uav_sensing_range * 3:  # broader sensing for AoI heatmap
                gx = int(poi.x / cell_size)
                gy = int(poi.y / cell_size)
                gx = min(max(gx, 0), res - 1)
                gy = min(max(gy, 0), res - 1)
                # Normalize AoI
                grid[gx, gy] = min(poi.aoi / self.cfg.surv_aoi_threshold, 2.0)

        return grid

    def _get_info(self) -> dict:
        """Compute metrics."""
        # Valid handling ratio for surveillance (Eq. 6)
        surv_valid = sum(
            1 for p in self.surv_pois if p.aoi <= p.aoi_threshold
        )
        I_surv = surv_valid / max(len(self.surv_pois), 1)

        # Valid handling ratio for emergency
        total_emer = len(self.emer_pois)
        if total_emer > 0:
            emer_valid = sum(
                1 for p in self.emer_pois
                if p.handled and (p.aoi <= p.aoi_threshold or not p.active)
            )
            I_emer = emer_valid / total_emer
        else:
            I_emer = 1.0

        # Energy consumption ratio (Section III-D)
        total_energy_used = sum(
            self.cfg.max_energy - u.energy for u in self.uavs
        )
        eta = total_energy_used / (self.cfg.num_uavs * self.cfg.max_energy * self.current_timeslots + 1e-8)
        eta = max(eta, 1e-8)

        # Valid task handling index (Eq. 7)
        I_index = min(I_emer, I_surv) / eta

        return {
            'I_surv': I_surv,
            'I_emer': I_emer,
            'I_index': I_index,
            'energy_ratio': eta,
            'surv_handled': self._prev_surv_handled,
            'emer_handled': self._prev_emer_handled,
            'timeslot': self.current_timeslot,
        }

    @property
    def current_timeslots(self):
        return self.current_timeslot

    def get_new_emergency_pois(self) -> List[PoI]:
        """Get PoIs that were just created and need assignment."""
        pois = []
        for poi_id in self._emer_just_assigned:
            for p in self.emer_pois:
                if p.poi_id == poi_id:
                    pois.append(p)
        self._emer_just_assigned = []
        return pois

    def get_uav_positions(self) -> np.ndarray:
        """Get all UAV positions as (U, 2) array."""
        return np.array([[u.x, u.y] for u in self.uavs])

    def render_ascii(self) -> str:
        """Simple ASCII visualization of the environment."""
        lines = [f"=== Timeslot {self.current_timeslot}/{self.cfg.max_timeslots} ==="]
        for uav in self.uavs:
            lines.append(f"  UAV {uav.uav_id}: pos=({uav.x:.0f},{uav.y:.0f}) E={uav.energy:.0f}")
        active_emer = [p for p in self.emer_pois if p.active]
        lines.append(f"  Active emergencies: {len(active_emer)}")
        for p in active_emer[:5]:
            lines.append(f"    PoI {p.poi_id}: ({p.x:.0f},{p.y:.0f}) AoI={p.aoi}/{p.aoi_threshold}")
        return "\n".join(lines)
