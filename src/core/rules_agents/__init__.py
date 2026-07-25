import importlib
import glob
import os
import sys

# Ensure vendor/cg-lib is in sys.path so rule-based agents can import cg.api
vendor_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "cg-lib")
vendor_path = os.path.abspath(vendor_path)
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

def get_rules_agent(agent_name: str):
    """
    Dynamically loads and returns the 'agent' function from the specified rules-based agent module.
    """
    module_name = f"src.core.rules_agents.{agent_name}"
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "agent"):
            return module.agent
        else:
            raise ValueError(f"Agent module {agent_name} does not contain an 'agent' function.")
    except ImportError as e:
        raise ImportError(f"Failed to load rules agent '{agent_name}': {e}")

def get_available_rules_agents():
    """
    Returns a list of all available rules-based agents.
    """
    agents_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(agents_dir, "*.py"))
    agents = []
    for f in files:
        basename = os.path.basename(f)
        if basename != "__init__.py":
            agents.append(basename.replace(".py", ""))
    return agents
