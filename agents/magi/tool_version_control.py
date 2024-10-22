import os
import git
from typing import Dict, Any

class ToolVersionControl:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
            self.repo = git.Repo.init(repo_path)
        else:
            self.repo = git.Repo(repo_path)

    def commit_tool(self, tool_name: str, tool_data: Dict[str, Any], message: str):
        file_path = os.path.join(self.repo_path, f"{tool_name}.json")
        with open(file_path, "w") as f:
            json.dump(tool_data, f, indent=2)
        self.repo.index.add([file_path])
        self.repo.index.commit(message)

    def get_tool_version(self, tool_name: str, version: str = "HEAD") -> Dict[str, Any]:
        file_path = os.path.join(self.repo_path, f"{tool_name}.json")
        try:
            tool_data = self.repo.git.show(f"{version}:{file_path}")
            return json.loads(tool_data)
        except git.exc.GitCommandError:
            return None

    def list_tool_versions(self, tool_name: str) -> List[str]:
        file_path = os.path.join(self.repo_path, f"{tool_name}.json")
        versions = []
        for commit in self.repo.iter_commits(paths=[file_path]):
            versions.append(commit.hexsha)
        return versions

    def create_branch(self, branch_name: str):
        self.repo.git.checkout("-b", branch_name)

    def switch_branch(self, branch_name: str):
        self.repo.git.checkout(branch_name)

    def merge_branch(self, source_branch: str, target_branch: str):
        self.switch_branch(target_branch)
        self.repo.git.merge(source_branch)

    def delete_branch(self, branch_name: str):
        self.repo.git.branch("-d", branch_name)
