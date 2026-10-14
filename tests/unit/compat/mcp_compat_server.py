"""Stdio MCP server the compat fixtures reference as `compat`."""

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x, which generated the fixtures
    from mcp.server.fastmcp import FastMCP as MCPServer  # pyright: ignore[reportAttributeAccessIssue]

server = MCPServer("compat")


@server.tool()
def shout(text: str) -> str:
    """Return text uppercased with an exclamation mark."""
    return text.upper() + "!"


if __name__ == "__main__":
    server.run()
