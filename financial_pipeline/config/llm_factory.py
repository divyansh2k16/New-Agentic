"""
LLM Factory

Returns a LangChain chat model configured for the active provider.
Set LLM_PROVIDER=openai in .env to use OpenAI instead of Anthropic.
"""
from config.settings import get_settings


def get_llm(model_tier: str = "primary", max_tokens: int = 1024):
    """
    Return the appropriate LangChain chat model.

    Args:
        model_tier: "primary" (more capable) or "fast" (cheaper/quicker)
        max_tokens: Maximum tokens for the response

    Returns:
        A LangChain BaseChatModel instance (ChatAnthropic or ChatOpenAI)
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = (
            settings.openai_primary_model
            if model_tier == "primary"
            else settings.openai_fast_model
        )
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            max_tokens=max_tokens,
        )

    # Default: Anthropic
    from langchain_anthropic import ChatAnthropic
    model = (
        settings.primary_llm_model
        if model_tier == "primary"
        else settings.fast_llm_model
    )
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
    )
