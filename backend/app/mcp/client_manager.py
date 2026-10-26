"""Official MCP SDK stdio client manager for configured MCP servers.

The prototype defaults to deterministic in-process adapters. Set MCP_SERVERS to
JSON mapping server names to command/args to connect to real stdio servers.
"""
import json, os
from contextlib import AsyncExitStack
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientManager:
    def __init__(self) -> None:
        self.stack=AsyncExitStack(); self.sessions:dict[str,ClientSession]={}

    async def connect_configured(self) -> None:
        config=json.loads(os.getenv("MCP_SERVERS","{}"))
        for name,spec in config.items():
            params=StdioServerParameters(command=spec["command"],args=spec.get("args",[]),env=spec.get("env"))
            read,write=await self.stack.enter_async_context(stdio_client(params))
            session=await self.stack.enter_async_context(ClientSession(read,write))
            await session.initialize(); self.sessions[name]=session

    async def list_tools(self) -> list[dict[str,Any]]:
        tools=[]
        for server,session in self.sessions.items():
            response=await session.list_tools()
            tools.extend({"name":f"{server}.{t.name}","description":t.description,"inputSchema":t.inputSchema} for t in response.tools)
        return tools

    async def call_tool(self,qualified_name:str,arguments:dict[str,Any]) -> Any:
        server,name=qualified_name.split(".",1)
        result=await self.sessions[server].call_tool(name,arguments)
        if result.isError: raise RuntimeError(f"MCP tool {qualified_name} returned an error")
        return result

    async def close(self) -> None:
        await self.stack.aclose(); self.sessions.clear()
