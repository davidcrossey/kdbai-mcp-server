# This file implements Embeddings Provider classes

import asyncio
import logging
from typing import Dict, Type
from collections import Counter
from functools import lru_cache
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---- Base Embedding Provider Interface ----
class EmbeddingProvider(ABC):
    @abstractmethod
    async def dense_embed(self, text: str, model_name: str) -> list[float]:
        """
        Encode text using the specified model.

        Args:
            text: Text to encode
            model_name: Specific model to use

        Returns:
           List[float]: Text embedding as a list of floats
        """

        pass

    @abstractmethod
    async def sparse_embed(self, text: str, model_name: str) -> Dict[str, int]:
        """
        Generate sparse embeddings (token counts) for text using the specified model.

        Args:
            text: Text to encode
            model_name: Specific model to use

        Returns:
           Dict[str, int]: Dictionary mapping token IDs to their counts
        """
        pass

    def cleanup_embedding_model(self):
        # Cleanup model from cache. Override if concrete provider is using lru cache.
        pass

# ---- Registry ----
PROVIDER_REGISTRY: Dict[str, Type[EmbeddingProvider]] = {}

def register_provider(name: str):
    def wrapper(cls):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return wrapper

# ---- Provider Factory ----
def get_provider(name: str) -> EmbeddingProvider:
    cls = PROVIDER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown provider: {name}")
    return cls()



#----------------------------------------------------------------------#
#   Implementation of Embedding Providers
#----------------------------------------------------------------------#
@register_provider("openai")
class OpenAIProvider(EmbeddingProvider):
    @lru_cache()
    def get_model(self):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Add it in the pyproject.toml")

        logger.info(f"Setting up OpenAI client")
        return AsyncOpenAI()  # User should configure API key via environment

    # dense_embed implementation
    async def dense_embed(self, text: str, model_name: str) -> list[float]:
        model = self.get_model()
        embedding = (await model.embeddings.create(
                model=model_name,
                input=text
            )).data[0].embedding

        return embedding

    # sparse_embed implementation
    async def sparse_embed(self, text: str, model_name: str) -> Dict[str, int]:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken required for OpenAI sparse embeddings is not installed."
                "Add it in the pyproject.toml"
            )

        def tokenize_and_count():
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"Unknown model {model_name}")
            token_ids = encoding.encode(text)
            return dict(Counter(token_ids))

        token_counts = await asyncio.to_thread(tokenize_and_count)
        return token_counts

    # override cleanup function for lru_cache usage
    def cleanup_embedding_model(self):
        return self.get_model().cache_clear()


@register_provider("fastembed")
class FastEmbedProvider(EmbeddingProvider):
    @lru_cache()
    def get_dense_model(self, model_name: str):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed not installed. Add it in the pyproject.toml")
        logger.info(f"Loading FastEmbed dense model: {model_name}")
        return TextEmbedding(model_name=model_name)

    @lru_cache()
    def get_sparse_model(self, model_name: str):
        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise ImportError("fastembed not installed. Add it in the pyproject.toml")
        logger.info(f"Loading FastEmbed sparse model: {model_name}")
        return SparseTextEmbedding(model_name=model_name)

    def _query_text(self, text: str, model_name: str) -> str:
        if "multilingual-e5" in model_name or model_name.startswith("intfloat/e5"):
            return f"query: {text}"
        return text

    async def dense_embed(self, text: str, model_name: str) -> list[float]:
        model = self.get_dense_model(model_name)
        embedding = await asyncio.to_thread(lambda: next(model.embed([self._query_text(text, model_name)])))
        return embedding.tolist()

    async def sparse_embed(self, text: str, model_name: str) -> Dict[str, int]:
        model = self.get_sparse_model(model_name)
        result = await asyncio.to_thread(lambda: next(model.embed([self._query_text(text, model_name)])))
        return {str(int(idx)): int(val) for idx, val in zip(result.indices, result.values)}

    def cleanup_embedding_model(self):
        self.get_dense_model.cache_clear()
        self.get_sparse_model.cache_clear()


@register_provider("sentence_transformers")
class SentenceTransformerProvider(EmbeddingProvider):
    @lru_cache()
    def get_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence_transformers not installed. Add it in the pyproject.toml")

        logger.info(f"Loading SentenceTransformer model: {model_name}")

        kwargs = {}
        return SentenceTransformer(model_name, **kwargs)

    # dense_embed implementation
    async def dense_embed(self, text: str, model_name: str) -> list[float]:
        model = self.get_model(model_name)
        embedding = await asyncio.to_thread(model.encode, text)
        return embedding.tolist()

    # sparse_embed implementation
    async def sparse_embed(self, text: str, model_name: str) -> Dict[str, int]:
        def tokenize_and_count():
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                tokens = model.tokenizer.tokenize(text)
                token_ids = model.tokenizer.convert_tokens_to_ids(tokens)
                return dict(Counter(token_ids))
        model = self.get_model(model_name)
        token_counts = await asyncio.to_thread(tokenize_and_count)
        return token_counts

    # override cleanup function for lru_cache usage
    def cleanup_embedding_model(self):
        return self.get_model().cache_clear()

