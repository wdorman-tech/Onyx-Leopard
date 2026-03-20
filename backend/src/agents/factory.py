from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from src.config import settings


def create_sonnet():
    return ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type=ModelType.CLAUDE_3_5_SONNET,
        api_key=settings.anthropic_api_key,
    )


def create_haiku():
    return ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type=ModelType.CLAUDE_3_5_HAIKU,
        api_key=settings.anthropic_api_key,
    )
