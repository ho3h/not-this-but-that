"""Kill-the-AI-ism gauntlet — corpus harvesting and gauntlet machinery.

Submodules:
  forms             — Form enum, shared by both detectors.
  harvest_detector  — permissive F1-F8 miner. Intentionally noisy; FPs are
                       killed by hand-verification during corpus build.
                       NEVER used to score gauntlet attacks (anti-circularity
                       rule from operating_protocol.md §1.5 / §2.7).
  referee           — strict F1-F8 evaluation classifier. Validated against
                       a hand-labeled independent holdout. THIS is the only
                       thing that scores the gauntlet.
"""

from gauntlet.forms import CORE_FORMS, Form

__all__ = ["Form", "CORE_FORMS"]
