"""Startup state/KV split for hybrid models under a memory-owning speculative policy."""

# Policies that own the memory partition actuator.
_MEMORY_POLICIES = ("memory", "conditional")


def spec_state_memory_ratio(
    *,
    policy_name: str,
    base_ratio: float,
    slots_per_req: int,
    num_draft_tokens: int,
) -> float:
    """The state:KV memory ratio once each running request's verify states are counted.

    `base_ratio` balances the slots_per_req persistent state slots of a request
    against its KV; speculative verify adds num_draft_tokens intermediate states
    per request, so a memory-owning policy scales the state side by
    (slots_per_req + num_draft_tokens) / slots_per_req. Other policies keep it.
    """
    if policy_name not in _MEMORY_POLICIES:
        return base_ratio
    return base_ratio * (slots_per_req + num_draft_tokens) / slots_per_req
