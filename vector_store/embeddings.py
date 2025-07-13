"""
Vector embeddings module for Cricket-Insight Agent.

This module provides functionality for creating and managing embeddings
for intelligent tool selection and cricket analytics enhancement.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from cricket_agent.config import CricketConfig

logger = logging.getLogger(__name__)


class CricketEmbeddingsManager:
    """
    Manages vector embeddings for cricket analytics tools and queries.
    
    Provides intelligent tool selection based on semantic similarity
    between user queries and tool descriptions/capabilities.
    """

    def __init__(self, config: CricketConfig) -> None:
        """
        Initialize the embeddings manager.

        Args:
            config (CricketConfig): Configuration object with vector store settings.
        """
        self.config = config
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight, fast model
        self.collection_name = "cricket_tools"
        
        # Initialize sentence transformer
        self.encoder: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Tool embeddings cache
        self._tool_embeddings: Dict[str, np.ndarray] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize the vector store and embedding model.
        
        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            logger.info("Initializing Cricket Embeddings Manager...")
            
            # Initialize sentence transformer
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            
            # Initialize Chroma client
            persist_directory = Path(self.config.vector_store_path)
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing Chroma client at: {persist_directory}")
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Cricket analytics tools and capabilities"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize tool embeddings
            await self._initialize_tool_embeddings()
            
            self._is_initialized = True
            logger.info("Cricket Embeddings Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings manager: {e}")
            raise RuntimeError(f"Embeddings initialization failed: {e}") from e

    async def _initialize_tool_embeddings(self) -> None:
        """
        Initialize embeddings for all available cricket analytics tools.
        """
        # Define cricket analytics tools with detailed descriptions
        cricket_tools = {
            "batting_analysis": {
                "name": "Batting Analysis Tool",
                "description": """
                Comprehensive batting statistics and performance analysis.
                Calculates batting averages, strike rates, boundary percentages,
                consistency metrics, and identifies batting roles (anchor, aggressor, finisher).
                Ideal for player performance evaluation, match preparation, and team selection.
                Keywords: runs, batting, average, strike rate, boundaries, consistency, innings
                """,
                "capabilities": [
                    "batting average calculation",
                    "strike rate analysis", 
                    "boundary percentage metrics",
                    "consistency index calculation",
                    "batting role identification",
                    "player performance tracking"
                ]
            },
            "bowling_analysis": {
                "name": "Bowling Analysis Tool", 
                "description": """
                Detailed bowling performance analysis and statistics.
                Computes bowling averages, economy rates, strike rates, wicket patterns,
                and bowling impact metrics. Determines bowling roles and effectiveness.
                Perfect for bowling strategy, opposition analysis, and performance reviews.
                Keywords: wickets, bowling, economy, average, strike rate, figures, overs
                """,
                "capabilities": [
                    "bowling average calculation",
                    "economy rate analysis",
                    "bowling strike rate metrics", 
                    "wicket pattern analysis",
                    "bowling impact assessment",
                    "bowling role classification"
                ]
            },
            "match_analysis": {
                "name": "Match Analysis Tool",
                "description": """
                In-depth match dynamics and comparative analysis.
                Analyzes innings comparisons, match momentum shifts, partnership contributions,
                turning points, and win probability calculations. Essential for match reviews,
                tactical analysis, and strategic planning.
                Keywords: match, innings, momentum, partnerships, comparison, tactics, strategy
                """,
                "capabilities": [
                    "innings comparison analysis",
                    "match momentum tracking",
                    "partnership analysis",
                    "turning point identification", 
                    "win probability calculation",
                    "match phase analysis"
                ]
            },
            "team_performance": {
                "name": "Team Performance Tool",
                "description": """
                Comprehensive team-level statistics and performance tracking.
                Evaluates team strengths, weaknesses, win-loss records, head-to-head comparisons,
                and overall team effectiveness. Crucial for team management, recruitment,
                and competitive analysis.
                Keywords: team, performance, statistics, comparison, strengths, weaknesses, records
                """,
                "capabilities": [
                    "team statistics compilation",
                    "head-to-head comparisons", 
                    "win-loss record analysis",
                    "team strength assessment",
                    "weakness identification",
                    "competitive benchmarking"
                ]
            }
        }
        
        # Create embeddings for each tool
        logger.info("Creating embeddings for cricket analytics tools...")
        
        documents = []
        metadatas = []
        ids = []
        
        for tool_id, tool_info in cricket_tools.items():
            # Combine description and capabilities for rich context
            full_description = f"{tool_info['description']} Capabilities: {', '.join(tool_info['capabilities'])}"
            
            documents.append(full_description)
            metadatas.append({
                "tool_id": tool_id,
                "name": tool_info["name"],
                "type": "cricket_analytics_tool"
            })
            ids.append(tool_id)
        
        # Add documents to collection
        if documents:
            # Check if tools already exist
            existing_count = self.collection.count()
            if existing_count == 0:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} cricket tools to vector store")
            else:
                logger.info(f"Vector store already contains {existing_count} items")

    async def find_relevant_tools(
        self, 
        query: str, 
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant cricket analytics tools for a given query.

        Args:
            query (str): User query or question.
            top_k (int): Maximum number of tools to return.
            similarity_threshold (float): Minimum similarity score threshold.

        Returns:
            List[Dict[str, Any]]: List of relevant tools with metadata and scores.
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            logger.debug(f"Finding relevant tools for query: '{query}'")
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            relevant_tools = []
            
            if results["ids"] and results["ids"][0]:
                for i, tool_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= similarity_threshold:
                        tool_info = {
                            "tool_id": tool_id,
                            "similarity_score": round(similarity, 3),
                            "metadata": results["metadatas"][0][i],
                            "description": results["documents"][0][i][:200] + "..."
                        }
                        relevant_tools.append(tool_info)
                        logger.debug(f"Found relevant tool: {tool_id} (similarity: {similarity:.3f})")
            
            logger.info(f"Found {len(relevant_tools)} relevant tools for query")
            return relevant_tools
            
        except Exception as e:
            logger.error(f"Error finding relevant tools: {e}")
            return []

    async def add_query_feedback(
        self, 
        query: str, 
        selected_tools: List[str],
        feedback_score: float
    ) -> None:
        """
        Add user feedback to improve tool selection over time.

        Args:
            query (str): Original user query.
            selected_tools (List[str]): Tools that were selected/used.
            feedback_score (float): User satisfaction score (0.0 to 1.0).
        """
        try:
            logger.debug(f"Adding feedback for query: '{query}' with score: {feedback_score}")
            
            # Create feedback document
            feedback_doc = f"Query: {query} | Selected tools: {', '.join(selected_tools)} | Score: {feedback_score}"
            
            # Add to feedback collection (create if needed)
            try:
                feedback_collection = self.chroma_client.get_collection("cricket_feedback")
            except ValueError:
                feedback_collection = self.chroma_client.create_collection(
                    name="cricket_feedback",
                    metadata={"description": "User feedback for tool selection"}
                )
            
            # Add feedback document
            feedback_id = f"feedback_{len(query)}_{hash(query) % 10000}"
            feedback_collection.add(
                documents=[feedback_doc],
                metadatas=[{
                    "query": query,
                    "tools": ",".join(selected_tools),
                    "score": feedback_score,
                    "type": "user_feedback"
                }],
                ids=[feedback_id]
            )
            
            logger.info("Successfully added user feedback")
            
        except Exception as e:
            logger.error(f"Error adding query feedback: {e}")

    async def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for cricket analytics tools.

        Returns:
            Dict[str, Any]: Usage statistics and insights.
        """
        try:
            if not self._is_initialized:
                await self.initialize()
            
            stats = {
                "total_tools": self.collection.count(),
                "vector_store_path": self.config.vector_store_path,
                "model_name": self.model_name,
                "collection_name": self.collection_name
            }
            
            # Get feedback stats if available
            try:
                feedback_collection = self.chroma_client.get_collection("cricket_feedback")
                stats["feedback_entries"] = feedback_collection.count()
            except ValueError:
                stats["feedback_entries"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}

    async def reset_vector_store(self) -> None:
        """
        Reset the vector store (useful for development/testing).
        
        Warning: This will delete all stored embeddings and feedback.
        """
        try:
            logger.warning("Resetting vector store...")
            
            if self.chroma_client:
                # Delete collections
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info(f"Deleted collection: {self.collection_name}")
                except ValueError:
                    pass
                
                try:
                    self.chroma_client.delete_collection("cricket_feedback")
                    logger.info("Deleted feedback collection")
                except ValueError:
                    pass
            
            # Reinitialize
            self._is_initialized = False
            await self.initialize()
            
            logger.info("Vector store reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            raise


# Utility functions for embeddings management

async def create_embeddings_manager(config: CricketConfig) -> CricketEmbeddingsManager:
    """
    Factory function to create and initialize an embeddings manager.

    Args:
        config (CricketConfig): Configuration object.

    Returns:
        CricketEmbeddingsManager: Initialized embeddings manager.
    """
    manager = CricketEmbeddingsManager(config)
    
    if config.enable_vector_store:
        await manager.initialize()
    
    return manager


async def test_embeddings_setup() -> None:
    """
    Test function to verify embeddings setup is working correctly.
    """
    from cricket_agent.config import CricketConfig
    
    # Create test config
    config = CricketConfig(
        enable_vector_store=True,
        vector_store_path="./test_vector_store"
    )
    
    # Initialize manager
    manager = await create_embeddings_manager(config)
    
    # Test queries
    test_queries = [
        "What is Virat Kohli's batting average?",
        "How is the bowling economy rate calculated?", 
        "Compare team performance between India and Australia",
        "Analyze the match momentum in the last game"
    ]
    
    print("Testing cricket embeddings manager...")
    for query in test_queries:
        tools = await manager.find_relevant_tools(query, top_k=2)
        print(f"\nQuery: {query}")
        for tool in tools:
            print(f"  - {tool['metadata']['name']} (score: {tool['similarity_score']})")
    
    # Get stats
    stats = await manager.get_tool_usage_stats()
    print(f"\nVector store stats: {stats}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_embeddings_setup())