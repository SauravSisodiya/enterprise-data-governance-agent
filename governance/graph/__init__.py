"""
LangGraph orchestration.

The graph does not reimplement the pipeline. It rewires the SAME node functions
that the sequential runner calls, from governance/graph/nodes.py. Parity between
the two is therefore structural rather than something that has to be verified by
comparing outputs - there is only one implementation of each step.

    metadata_node
         |
    +----+----+           quality and compliance both consume the catalog and
    |         |           neither consumes the other, so they fan out here
 quality  compliance
    |         |           both append to `issues`, which needs its
    +----+----+           operator.add reducer or LangGraph raises
         |                InvalidUpdateError
     join_node
         |
     risk_node            scores and ranks
         |
     cite_node            attaches clause text
         |
   recommend_node         builds remediation from the playbook
         |
    review_gate           High and Critical wait for a human
         |
    narrative_node        optional prose; skipped when no model is reachable
"""
from governance.graph.build import build_graph, run_graph

__all__ = ["build_graph", "run_graph"]
