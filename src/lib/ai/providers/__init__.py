from .gemini import gemini_provider
from .groq import groq_provider
from .ollama import ollama_provider
from .openrouter import openrouter_provider

REGISTRY = {
    "ollama": ollama_provider,
    "groq": groq_provider,
    "gemini": gemini_provider,
    "openrouter": openrouter_provider,
}
