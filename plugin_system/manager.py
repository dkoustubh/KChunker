import importlib
import pkgutil
from pathlib import Path
from typing import TypeVar

from plugin_system.base import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseParser,
    BasePlugin,
    BaseVectorStore,
)
from utils.logging import logger

T = TypeVar("T", bound=BasePlugin)


class PluginManager:
    """Manages system plugins including dynamic loading, registration, and lookup."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}
        self._chunkers: dict[str, BaseChunker] = {}
        self._embeddings: dict[str, BaseEmbeddingModel] = {}
        self._vector_stores: dict[str, BaseVectorStore] = {}

    def register_parser(self, parser: BaseParser) -> None:
        """Registers a parser plugin."""
        logger.info(
            "Registering parser plugin", name=parser.name, version=parser.version
        )
        parser.initialize()
        self._parsers[parser.name] = parser

    def register_chunker(self, chunker: BaseChunker) -> None:
        """Registers a chunker plugin."""
        logger.info(
            "Registering chunker plugin", name=chunker.name, version=chunker.version
        )
        chunker.initialize()
        self._chunkers[chunker.name] = chunker

    def register_embedding_model(self, embedding_model: BaseEmbeddingModel) -> None:
        """Registers an embedding model plugin."""
        logger.info(
            "Registering embedding model plugin",
            name=embedding_model.name,
            version=embedding_model.version,
        )
        embedding_model.initialize()
        self._embeddings[embedding_model.name] = embedding_model

    def register_vector_store(self, vector_store: BaseVectorStore) -> None:
        """Registers a vector store plugin."""
        logger.info(
            "Registering vector store plugin",
            name=vector_store.name,
            version=vector_store.version,
        )
        vector_store.initialize()
        self._vector_stores[vector_store.name] = vector_store

    def get_parser(self, name: str) -> BaseParser | None:
        """Retrieves a parser by name."""
        return self._parsers.get(name)

    def get_all_parsers(self) -> list[BaseParser]:
        """Retrieves all registered parsers."""
        return list(self._parsers.values())

    def get_chunker(self, name: str) -> BaseChunker | None:
        """Retrieves a chunker by name."""
        return self._chunkers.get(name)

    def get_all_chunkers(self) -> list[BaseChunker]:
        """Retrieves all registered chunkers."""
        return list(self._chunkers.values())

    def get_embedding_model(self, name: str) -> BaseEmbeddingModel | None:
        """Retrieves an embedding model by name."""
        return self._embeddings.get(name)

    def get_vector_store(self, name: str) -> BaseVectorStore | None:
        """Retrieves a vector store by name."""
        return self._vector_stores.get(name)

    def discover_plugins(self, package_path: str) -> None:
        """Dynamically imports all submodules in a package to trigger decorators."""
        try:
            package = importlib.import_module(package_path)
            if not package.__file__:
                return
            package_dir = Path(package.__file__).parent
            for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
                importlib.import_module(f"{package_path}.{module_name}")
        except Exception as e:
            logger.error(
                "Failed to dynamically discover plugins",
                package=package_path,
                error=str(e),
            )

    def shutdown(self) -> None:
        """Calls cleanup routines on all registered plugins."""
        for name, parser in self._parsers.items():
            try:
                parser.shutdown()
            except Exception as e:
                logger.error("Error shutting down parser", name=name, error=str(e))

        for name, chunker in self._chunkers.items():
            try:
                chunker.shutdown()
            except Exception as e:
                logger.error("Error shutting down chunker", name=name, error=str(e))

        for name, model in self._embeddings.items():
            try:
                model.shutdown()
            except Exception as e:
                logger.error(
                    "Error shutting down embedding model", name=name, error=str(e)
                )

        for name, store in self._vector_stores.items():
            try:
                store.shutdown()
            except Exception as e:
                logger.error(
                    "Error shutting down vector store", name=name, error=str(e)
                )


plugin_manager = PluginManager()
