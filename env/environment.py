"""
Disaster Response Coordination System — Main OpenEnv Interface
==============================================================
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

from env.models import ActionWrapper, EpisodeResult, Observation, Reward
from env.tasks import Task1Environment, Task2Environment, Task3Environment
from env.graders import Task1Grader, Task2Grader, Task3Grader

TASK_REGISTRY = {
    "task1_prioritization":       (Task1Environment, Task1Grader),
    "task2_resource_allocation":  (Task2Environment, Task2Grader),
    "task3_dynamic_coordination": (Task3Environment, Task3Grader),
}
VALID_TASKS = list(TASK_REGISTRY.keys())


class DisasterResponseEnv:
    def __init__(self, task_id: str = "task1_prioritization", seed: int = 42):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
        self._task_id  = task_id
        self._seed     = seed
        env_cls, grader_cls = TASK_REGISTRY[task_id]
        self._task_env = env_cls(seed=seed)
        self._grader   = grader_cls()
        self._obs: Optional[Observation] = None

    def reset(self) -> Observation:
        obs = self._task_env.reset()
        self._obs = obs
        return obs

    def step(self, action: Union[ActionWrapper, Dict[str, Any]]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            action = ActionWrapper(**action)
        obs, reward, done, info = self._task_env.step(action)
        self._obs = obs
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return self._task_env.state()

    def grade(self) -> EpisodeResult:
        result = self._task_env.grade()
        result.grader_score = self._grader.grade(result)
        return result

    def grade_description(self) -> str:
        return self._grader.describe(self._task_env.grade())

    def render(self) -> str:
        if self._obs is None:
            return "Not initialized. Call reset() first."
        return self._obs.to_text()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def valid_tasks(self) -> List[str]:
        return VALID_TASKS

    def observation_schema(self) -> Dict[str, Any]:
        return Observation.model_json_schema()

    def action_schema(self) -> Dict[str, Any]:
        return ActionWrapper.model_json_schema()

    def reward_schema(self) -> Dict[str, Any]:
        return Reward.model_json_schema()
