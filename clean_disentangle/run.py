"""Compatibility entrypoint for the relocated Stage1 trainer.

New code should import or execute ``clean_disentangle.stage1.train_stage1``.
"""

from .stage1.train_stage1 import *  # noqa: F401,F403
from .stage1.train_stage1 import main


if __name__ == "__main__":
    main()
