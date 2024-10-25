"""GitHub API integration for MAGI."""

import os
import aiohttp
import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GitHubRepo:
    """Represents a GitHub repository."""
    name: str
    full_name: str
    description: str
    url: str
    clone_url: str
    stars: int
    forks: int
    last_updated: datetime
    topics: List[str]
    language: str

class GitHubClient:
    """
    Client for interacting with GitHub's REST API.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('GITHUB_API_TOKEN')
        if not self.api_token:
            raise ValueError("GitHub API token not provided")
            
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.api_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    async def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        sort: str = "stars",
        order: str = "desc",
        max_results: int = 10
    ) -> List[GitHubRepo]:
        """
        Search GitHub repositories.

        :param query: Search query
        :param language: Filter by programming language
        :param sort: Sort by (stars, forks, updated)
        :param order: Sort order (asc, desc)
        :param max_results: Maximum number of results
        :return: List of matching repositories
        """
        search_query = query
        if language:
            search_query += f" language:{language}"
            
        params = {
            "q": search_query,
            "sort": sort,
            "order": order,
            "per_page": max_results
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/search/repositories",
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"GitHub API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                
                return [
                    GitHubRepo(
                        name=repo['name'],
                        full_name=repo['full_name'],
                        description=repo['description'] or "",
                        url=repo['html_url'],
                        clone_url=repo['clone_url'],
                        stars=repo['stargazers_count'],
                        forks=repo['forks_count'],
                        last_updated=datetime.fromisoformat(
                            repo['updated_at'].replace('Z', '+00:00')),
                        topics=repo.get('topics', []),
                        language=repo['language'] or "Unknown"
                    )
                    for repo in data['items']
                ]

    async def search_code(
        self,
        query: str,
        language: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search code in GitHub repositories.

        :param query: Search query
        :param language: Filter by programming language
        :param max_results: Maximum number of results
        :return: List of matching code files
        """
        search_query = query
        if language:
            search_query += f" language:{language}"
            
        params = {
            "q": search_query,
            "per_page": max_results
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/search/code",
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"GitHub API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                results = []
                
                for item in data['items']:
                    # Get file content
                    content = await self.get_file_content(
                        item['repository']['full_name'],
                        item['path']
                    )
                    
                    results.append({
                        'repository': item['repository']['full_name'],
                        'path': item['path'],
                        'url': item['html_url'],
                        'content': content
                    })
                    
                return results

    async def get_file_content(
        self,
        repo_full_name: str,
        file_path: str
    ) -> str:
        """
        Get content of a file from a repository.

        :param repo_full_name: Full repository name (owner/repo)
        :param file_path: Path to file in repository
        :return: File content
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/repos/{repo_full_name}/contents/{file_path}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"GitHub API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                content = base64.b64decode(data['content']).decode('utf-8')
                return content

    async def clone_repository(
        self,
        repo_full_name: str,
        local_path: str,
        branch: str = "main"
    ) -> str:
        """
        Clone a repository to local path.

        :param repo_full_name: Full repository name (owner/repo)
        :param local_path: Local path to clone to
        :param branch: Branch to clone
        :return: Path to cloned repository
        """
        # Get repository details
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/repos/{repo_full_name}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"GitHub API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                clone_url = data['clone_url']
                
        # Clone repository
        clone_command = f"git clone -b {branch} {clone_url} {local_path}"
        process = await asyncio.create_subprocess_shell(
            clone_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Git clone failed: {stderr.decode()}")
            
        return local_path

    async def download_file(
        self,
        repo_full_name: str,
        file_path: str,
        local_path: str
    ) -> str:
        """
        Download a single file from a repository.

        :param repo_full_name: Full repository name (owner/repo)
        :param file_path: Path to file in repository
        :param local_path: Local path to save file
        :return: Path to downloaded file
        """
        content = await self.get_file_content(repo_full_name, file_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Write content to file
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return local_path
