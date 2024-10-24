import os
import json
from typing import Dict, Any
from agents.magi.tools.tool_version_control import ToolVersionControl

class ToolPersistence:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.version_control = ToolVersionControl(os.path.join(storage_dir, "versions"))

    def save_tool(self, name: str, code: str, description: str, parameters: Dict[str, Any]):
        tool_data = {
            "name": name,
            "code": code,
            "description": description,
            "parameters": parameters
        }
        file_path = os.path.join(self.storage_dir, f"{name}.json")
        with open(file_path, "w") as f:
            json.dump(tool_data, f, indent=2)
        self.version_control.commit_tool(name, tool_data, f"Add tool '{name}'")

    def load_tool(self, name: str, version: str = "HEAD") -> Dict[str, Any]:
        tool_data = self.version_control.get_tool_version(name, version)
        if tool_data:
            return tool_data
        else:
            file_path = os.path.join(self.storage_dir, f"{name}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    tool_data = json.load(f)
                return tool_data
            else:
                return None

    def load_all_tools(self, version: str = "HEAD") -> Dict[str, Dict[str, Any]]:
        tools = {}
        for file_name in os.listdir(self.storage_dir):
            if file_name.endswith(".json"):
                tool_name = file_name[:-5]  # Remove ".json" extension
                tool_data = self.load_tool(tool_name, version)
                if tool_data:
                    tools[tool_name] = tool_data
        return tools

    def update_tool(self, name: str, code: str, description: str, parameters: Dict[str, Any]):
        tool_data = {
            "name": name,
            "code": code,
            "description": description,
            "parameters": parameters
        }
        file_path = os.path.join(self.storage_dir, f"{name}.json")
        with open(file_path, "w") as f:
            json.dump(tool_data, f, indent=2)
        self.version_control.commit_tool(name, tool_data, f"Update tool '{name}'")

    def delete_tool(self, name: str):
        file_path = os.path.join(self.storage_dir, f"{name}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            self.version_control.commit_tool(name, {}, f"Delete tool '{name}'")

    def list_tool_versions(self, name: str) -> List[str]:
        return self.version_control.list_tool_versions(name)

    def load_tool_version(self, name: str, version: str) -> Dict[str, Any]:
        return self.version_control.get_tool_version(name, version)

    def create_tool_branch(self, branch_name: str):
        self.version_control.create_branch(branch_name)

    def switch_tool_branch(self, branch_name: str):
        self.version_control.switch_branch(branch_name)

    def merge_tool_branches(self, source_branch: str, target_branch: str):
        self.version_control.merge_branch(source_branch, target_branch)

    def delete_tool_branch(self, branch_name: str):
        self.version_control.delete_branch(branch_name)
