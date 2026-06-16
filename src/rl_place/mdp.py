"""MDP namespace for the M0609 pre-attached carry-and-place task."""

# Isaac Lab built-in lift/task terms and generic env MDP terms.
from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

# Local custom terms.
from .perception import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
