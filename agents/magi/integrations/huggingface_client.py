"""Hugging Face Hub integration for MAGI."""

import os
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HFModel:
    """Represents a Hugging Face model."""
    id: str
    name: str
    description: str
    downloads: int
    likes: int
    tags: List[str]
    last_modified: datetime
    pipeline_tag: str
    model_type: str

class HuggingFaceClient:
    """
    Client for interacting with the Hugging Face Hub API.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        if not self.api_token:
            raise ValueError("Hugging Face API token not provided")
            
        self.base_url = "https://huggingface.co/api"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json"
        }

    async def search_models(
        self,
        query: str,
        filter_tags: Optional[List[str]] = None,
        sort: str = "downloads",
        direction: str = "desc",
        limit: int = 10
    ) -> List[HFModel]:
        """
        Search models on Hugging Face Hub.

        :param query: Search query
        :param filter_tags: List of tags to filter by
        :param sort: Sort by (downloads, likes, modified)
        :param direction: Sort direction (asc, desc)
        :param limit: Maximum number of results
        :return: List of matching models
        """
        params = {
            "search": query,
            "sort": sort,
            "direction": direction,
            "limit": limit
        }
        
        if filter_tags:
            params["filter"] = " ".join(filter_tags)
            
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models",
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                
                return [
                    HFModel(
                        id=model['modelId'],
                        name=model['modelId'].split('/')[-1],
                        description=model.get('description', ''),
                        downloads=model.get('downloads', 0),
                        likes=model.get('likes', 0),
                        tags=model.get('tags', []),
                        last_modified=datetime.fromisoformat(
                            model['lastModified'].replace('Z', '+00:00')),
                        pipeline_tag=model.get('pipeline_tag', ''),
                        model_type=model.get('model_type', '')
                    )
                    for model in data
                ]

    async def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        :param model_id: Model ID (e.g., 'username/model-name')
        :return: Model details
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models/{model_id}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                return await response.json()

    async def download_model(
        self,
        model_id: str,
        local_dir: str,
        revision: str = "main"
    ) -> str:
        """
        Download a model to local directory.

        :param model_id: Model ID (e.g., 'username/model-name')
        :param local_dir: Local directory to save model
        :param revision: Model revision/branch
        :return: Path to downloaded model
        """
        # Ensure directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Get model files list
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models/{model_id}/tree/{revision}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                files = await response.json()
                
                # Download each file
                for file in files:
                    file_path = os.path.join(local_dir, file['path'])
                    
                    # Create directories if needed
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Download file
                    async with session.get(
                        f"{self.base_url}/models/{model_id}/resolve/{revision}/{file['path']}"
                    ) as file_response:
                        if file_response.status != 200:
                            logger.warning(
                                f"Failed to download {file['path']}: {file_response.status}")
                            continue
                            
                        with open(file_path, 'wb') as f:
                            f.write(await file_response.read())
                            
        return local_dir

    async def get_model_card(self, model_id: str) -> str:
        """
        Get the model card (README.md) content.

        :param model_id: Model ID (e.g., 'username/model-name')
        :return: Model card content
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models/{model_id}/readme"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                data = await response.json()
                return data['content']

    async def get_model_files(
        self,
        model_id: str,
        revision: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Get list of files in a model repository.

        :param model_id: Model ID (e.g., 'username/model-name')
        :param revision: Model revision/branch
        :return: List of files
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models/{model_id}/tree/{revision}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                return await response.json()

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        local_path: str,
        revision: str = "main"
    ) -> str:
        """
        Download a single file from a model repository.

        :param model_id: Model ID (e.g., 'username/model-name')
        :param file_path: Path to file in repository
        :param local_path: Local path to save file
        :param revision: Model revision/branch
        :return: Path to downloaded file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(
                f"{self.base_url}/models/{model_id}/resolve/{revision}/{file_path}"
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Hugging Face API error: {response.status} - {await response.text()}")
                    
                with open(local_path, 'wb') as f:
                    f.write(await response.read())
                    
        return local_path
