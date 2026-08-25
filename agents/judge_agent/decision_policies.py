"""
HalluciGuard - Domain Decision Policies (Compatibility Layer)
Delegates to the canonical domain_policies.py module.
"""

from domain_policies import DomainPolicy, DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY

# Alias for backward compatibility
DEFAULT_POLICY_REGISTRY = DEFAULT_DOMAIN_REGISTRY
