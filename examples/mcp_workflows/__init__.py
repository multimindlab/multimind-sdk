"""Example workflows for the MCP system.

Showcases:
  * CI/CD automation
  * Code review automation
  * Documentation generation
  * Multi-platform issue management
  * Basic workflow with Slack / Jira integrations

The previous version of this module eagerly imported a bunch of submodules
that no longer exist (the files were moved under ``examples/mcp_workflows/examples/``
and ``examples/mcp_workflows/workflows/``), which broke ``import examples.mcp`` and
in turn made the test-collection step skip every test that imports anything
inside this package.

Example packages should be lightweight at import time; users should import
the specific example module directly, e.g.::

    from examples.mcp_workflows.examples.ci_cd_example import main
"""
