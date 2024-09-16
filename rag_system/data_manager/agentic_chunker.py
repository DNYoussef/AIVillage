import uuid
import json
import logging
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

from .logger import logger
from .config import config

load_dotenv()

class PlanAwareAgenticChunker:
    def __init__(self, llm: ChatOpenAI):
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.id_truncate_limit: int = 5
        self.generate_new_metadata_ind: bool = True
        self.llm = llm

    def add_propositions(self, propositions: List[str], current_plan: str) -> None:
        for proposition in propositions:
            self.add_proposition(proposition, current_plan)

    def add_proposition(self, proposition: str, current_plan: str) -> None:
        logger.info(f"Adding: '{proposition}'")

        if not self.chunks:
            logger.info("No chunks, creating a new one")
            self._create_new_chunk(proposition, current_plan)
            return

        chunk_result = self._find_relevant_chunk(proposition, current_plan)

        if chunk_result and chunk_result['chunk_id']:
            chunk_id = chunk_result['chunk_id']
            logger.info(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self._add_proposition_to_chunk(chunk_id, proposition, chunk_result, current_plan)
        else:
            logger.info("No chunks found")
            self._create_new_chunk(proposition, current_plan)

    def _add_proposition_to_chunk(self, chunk_id: str, proposition: str, chunk_result: Dict[str, Any], current_plan: str) -> None:
        chunk = self.chunks[chunk_id]
        chunk['propositions'].append(proposition)
        
        for key in ['importance', 'category', 'relationship']:
            if key not in chunk:
                chunk[key] = []
            chunk[key].append(chunk_result[key])

        if self.generate_new_metadata_ind:
            chunk['summary'] = self._update_chunk_summary(chunk, current_plan)
            chunk['title'] = self._update_chunk_title(chunk, current_plan)

    @lru_cache(maxsize=128)
    def _update_chunk_summary(self, chunk: Dict[str, Any], current_plan: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
            A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.
            Consider the current plan when generating the summary.

            A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

            You will be given a group of propositions which are in the chunk, the chunk's current summary, and the current plan.

            Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
            Or month, generalize it to "date and times".

            Example:
            Input: Proposition: Greg likes to eat pizza
            Output: This chunk contains information about the types of food Greg likes to eat.

            Only respond with the chunk's new summary, nothing else.
            """),
            ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}\n\nCurrent plan:\n{current_plan}")
        ])

        runnable = PROMPT | self.llm
        return runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_plan": current_plan
        }).content

    @lru_cache(maxsize=128)
    def _update_chunk_title(self, chunk: Dict[str, Any], current_plan: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
            A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.
            Consider the current plan when generating the title.

            A good title will say what the chunk is about.

            You will be given a group of propositions which are in the chunk, chunk summary, the chunk title, and the current plan.

            Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
            Or month, generalize it to "date and times".

            Example:
            Input: Summary: This chunk is about dates and times that the author talks about
            Output: Date & Times

            Only respond with the new chunk title, nothing else.
            """),
            ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}\n\nCurrent plan:\n{current_plan}")
        ])

        runnable = PROMPT | self.llm
        return runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title'],
            "current_plan": current_plan
        }).content

    def _create_new_chunk(self, proposition: str, current_plan: str) -> None:
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition, current_plan)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary, current_plan)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks),
            'importance': [],
            'category': [],
            'relationships': []
        }
        logger.info(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    @lru_cache(maxsize=128)
    def _get_new_chunk_summary(self, proposition: str, current_plan: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
            You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.
            Consider the current plan when generating the summary.

            A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

            You will be given a proposition which will go into a new chunk and the current plan. This new chunk needs a summary.

            Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
            Or month, generalize it to "date and times".

            Example:
            Input: Proposition: Greg likes to eat pizza
            Output: This chunk contains information about the types of food Greg likes to eat.

            Only respond with the new chunk summary, nothing else.
            """),
            ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}\n\nCurrent plan:\n{current_plan}")
        ])

        runnable = PROMPT | self.llm
        return runnable.invoke({"proposition": proposition, "current_plan": current_plan}).content

    @lru_cache(maxsize=128)
    def _get_new_chunk_title(self, summary: str, current_plan: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
            You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.
            Consider the current plan when generating the title.

            A good chunk title is brief but encompasses what the chunk is about.

            You will be given a summary of a chunk which needs a title and the current plan.

            Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
            Or month, generalize it to "date and times".

            Example:
            Input: Summary: This chunk is about dates and times that the author talks about
            Output: Date & Times

            Only respond with the new chunk title, nothing else.
            """),
            ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}\n\nCurrent plan:\n{current_plan}")
        ])

        runnable = PROMPT | self.llm
        return runnable.invoke({"summary": summary, "current_plan": current_plan}).content

    def get_chunk_outline(self) -> str:
        return "\n\n".join([
            f"Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}"
            for chunk in self.chunks.values()
        ])

    def _find_relevant_chunk(self, proposition: str, current_plan: str) -> Optional[Dict[str, Any]]:
        current_chunk_outline = self.get_chunk_outline()
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Determine whether or not the "Proposition" should belong to any of the existing chunks.
            Consider the current plan when making this determination.

            A proposition should belong to a chunk if their meaning, direction, or intention are similar.
            The goal is to group similar propositions and chunks.

            If you think a proposition should be joined with a chunk, return the chunk id along with the following information:
            1. The importance of the proposition on a scale of 1 to 5 (5 being the highest).
            2. The category of the proposition (choose from: event, concept, place, object, document, organization, condition, misc).
            3. A brief (one or two sentence) description of how this proposition relates to the chunk.

            If you do not think an item should be joined with an existing chunk, just return "No chunks"

            Format your response as JSON with the following structure:
            {
                "chunk_id": "chunk_id or null if no chunks",
                "importance": 1-5,
                "category": "category",
                "relationship": "brief description of relationship"
            }

            Example:
            Input:
                - Proposition: "Greg really likes hamburgers"
                - Current Chunks:
                    - Chunk ID: 2n4l3d
                    - Chunk Name: Places in San Francisco
                    - Chunk Summary: Overview of the things to do with San Francisco Places

                    - Chunk ID: 93833k
                    - Chunk Name: Food Greg likes
                    - Chunk Summary: Lists of the food and dishes that Greg likes
            Output: 
            {
                "chunk_id": "93833k",
                "importance": 4,
                "category": "concept",
                "relationship": "This proposition adds to the list of foods Greg likes, specifically mentioning hamburgers."
            }
            """),
            ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
            ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}\n\nCurrent plan:\n{current_plan}")
        ])

        runnable = PROMPT | JsonOutputParser()
        try:
            result = runnable.invoke({
                "proposition": proposition,
                "current_chunk_outline": current_chunk_outline,
                "current_plan": current_plan
            })
            return result if result['chunk_id'] != "null" else None
        except Exception as e:
            logger.error(f"Error finding relevant chunk: {e}")
            return None

    def get_chunks(self, get_type: str = 'dict') -> Union[Dict[str, Dict[str, Any]], List[str]]:
        if get_type == 'dict':
            return self.chunks
        elif get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]
        else:
            raise ValueError("Invalid get_type. Use 'dict' or 'list_of_strings'.")

    def serialize_chunks(self) -> str:
        return json.dumps(self.chunks)

    def deserialize_chunks(self, serialized_chunks: str) -> None:
        self.chunks = json.loads(serialized_chunks)

    def pretty_print_chunks(self) -> None:
        logger.info(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk in self.chunks.values():
            logger.info(f"Chunk #{chunk['chunk_index']}")
            logger.info(f"Chunk ID: {chunk['chunk_id']}")
            logger.info(f"Title: {chunk['title']}")
            logger.info(f"Summary: {chunk['summary']}")
            logger.info("Propositions:")
            for i, prop in enumerate(chunk['propositions']):
                logger.info(f"    -{prop}")
                for key in ['importance', 'category', 'relationships']:
                    if i < len(chunk.get(key, [])):
                        logger.info(f"      {key.capitalize()}: {chunk[key][i]}")
            logger.info("\n")

    def pretty_print_chunk_outline(self) -> None:
        logger.info("Chunk Outline\n")
        logger.info(self.get_chunk_outline())