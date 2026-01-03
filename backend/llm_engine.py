"""
LLM Engine for image enrichment and caption generation.

Provides abstraction over different LLM providers (Ollama, Gemini)
for analyzing images and generating training captions.
"""
import asyncio
import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

# Settings and prompts paths
BACKEND_DIR = Path(__file__).parent
SETTINGS_PATH = BACKEND_DIR / "settings.json"
SETTINGS_TEMPLATE_PATH = BACKEND_DIR / "settings.template.json"
PROMPTS_PATH = BACKEND_DIR / "prompts.yaml"


@dataclass
class EnrichmentResult:
    """Result from LLM image enrichment."""
    shot_type: str = "unknown"
    pose_description: str = ""
    expression: str = ""
    clothing_description: str = ""
    lighting_description: str = ""
    background_description: str = ""
    full_description: str = ""
    quality_notes: Optional[str] = None
    captions: dict[str, str] = field(default_factory=dict)
    
    # Metadata
    provider: str = ""
    model: str = ""
    generated_at: Optional[datetime] = None
    error: Optional[str] = None


def load_settings() -> dict:
    """Load settings from JSON file."""
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, 'r') as f:
            return json.load(f)
    elif SETTINGS_TEMPLATE_PATH.exists():
        logger.warning("settings.json not found, using template defaults")
        with open(SETTINGS_TEMPLATE_PATH, 'r') as f:
            return json.load(f)
    else:
        logger.warning("No settings file found, using hardcoded defaults")
        return {
            "llm": {
                "provider": "ollama",
                "ollama": {"base_url": "http://localhost:11434", "model": "llava:13b"},
                "gemini": {"api_key": "", "model": "gemini-3-flash-preview"}
            },
            "processing": {"enable_enrichment": True, "caption_formats": ["SDXL", "Flux"]}
        }


def save_settings(settings: dict) -> None:
    """Save settings to JSON file."""
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(settings, f, indent=4)


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    if PROMPTS_PATH.exists():
        with open(PROMPTS_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning("prompts.yaml not found, using defaults")
        return {
            "enrichment": {
                "system": "You are an image analyst.",
                "analysis": "Analyze this image and return a JSON object."
            },
            "captions": {}
        }


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image path."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def enrich_image(
        self,
        image_path: str,
        reference_images: list[str],
        character_name: str,
        embedding_scores: dict,
        prompts: dict,
        caption_formats: list[str]
    ) -> EnrichmentResult:
        """
        Analyze an image and generate enrichment data.
        
        Args:
            image_path: Path to the target image to analyze
            reference_images: Paths to reference images for the character
            character_name: Name of the character
            embedding_scores: ML-derived scores (face_similarity, shot_type, etc.)
            prompts: Loaded prompts configuration
            caption_formats: List of caption formats to generate
            
        Returns:
            EnrichmentResult with analysis and captions
        """
        pass
    
    @abstractmethod
    async def check_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava:13b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    async def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    # Check if our model or a variant is available
                    return any(self.model in name or name in self.model for name in model_names)
                return False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    async def enrich_image(
        self,
        image_path: str,
        reference_images: list[str],
        character_name: str,
        embedding_scores: dict,
        prompts: dict,
        caption_formats: list[str]
    ) -> EnrichmentResult:
        """Analyze image using Ollama vision model."""
        result = EnrichmentResult(provider="ollama", model=self.model, generated_at=datetime.now())
        
        try:
            # Build the prompt
            system_prompt = prompts.get('enrichment', {}).get('system', '').format(
                character_name=character_name
            )
            analysis_prompt = prompts.get('enrichment', {}).get('analysis', '').format(
                character_name=character_name,
                face_similarity=embedding_scores.get('face_similarity', 'N/A'),
                shot_type=embedding_scores.get('shot_type', 'unknown')
            )
            
            # Prepare images - only target image for Ollama (multi-image not well supported by all models)
            # Reference context is provided in the prompt instead
            images = []
            if os.path.exists(image_path):
                images.append(image_to_base64(image_path))
            else:
                result.error = f"Image not found: {image_path}"
                return result
            
            # Call Ollama API (long timeout for slow local inference)
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"{system_prompt}\n\n{analysis_prompt}",
                        "images": images,
                        "stream": False
                        # Note: removed "format": "json" - not all vision models support it
                    }
                )
                
                if response.status_code != 200:
                    result.error = f"Ollama API error: {response.status_code}"
                    return result
                
                response_data = response.json()
                response_text = response_data.get('response', '')
                
            # Debug logging
            logger.info(f"Ollama raw response length: {len(response_text)} chars")
            if len(response_text) < 500:
                logger.info(f"Ollama response: {response_text}")
            else:
                logger.info(f"Ollama response (truncated): {response_text[:500]}...")
                
            # Parse JSON response - handle markdown code blocks
            try:
                clean_text = response_text.strip()
                # Remove markdown code blocks if present
                if clean_text.startswith('```'):
                    # Find the end of the first line (```json or ```)
                    first_newline = clean_text.find('\n')
                    if first_newline > 0:
                        clean_text = clean_text[first_newline + 1:]
                    # Remove trailing ```
                    if clean_text.endswith('```'):
                        clean_text = clean_text[:-3].strip()
                    elif '```' in clean_text:
                        clean_text = clean_text.rsplit('```', 1)[0].strip()
                
                analysis = json.loads(clean_text)
                result.shot_type = analysis.get('shot_type', 'unknown')
                result.pose_description = analysis.get('pose_description', '')
                result.expression = analysis.get('expression', '')
                result.clothing_description = analysis.get('clothing_description', '')
                result.lighting_description = analysis.get('lighting_description', '')
                result.background_description = analysis.get('background_description', '')
                result.full_description = analysis.get('full_description', '')
                result.quality_notes = analysis.get('quality_notes')
                logger.info(f"Parsed enrichment: shot_type={result.shot_type}, expression={result.expression[:30] if result.expression else 'None'}...")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Ollama JSON response: {e}")
                logger.warning(f"Response was: {response_text[:200] if response_text else 'EMPTY'}...")
                result.full_description = response_text
            
            # Generate captions for each format
            for format_name in caption_formats:
                caption_prompt = prompts.get('captions', {}).get(format_name)
                if caption_prompt:
                    caption = await self._generate_caption(
                        image_path, character_name, result, caption_prompt
                    )
                    result.captions[format_name] = caption
            
        except Exception as e:
            logger.exception(f"Ollama enrichment error: {e}")
            result.error = str(e)
        
        return result
    
    async def _generate_caption(
        self,
        image_path: str,
        character_name: str,
        analysis: EnrichmentResult,
        caption_prompt: str
    ) -> str:
        """Generate a caption for a specific format."""
        try:
            # Build context from analysis
            context = f"""
Based on this analysis of the image:
- Shot type: {analysis.shot_type}
- Pose: {analysis.pose_description}
- Expression: {analysis.expression}
- Clothing: {analysis.clothing_description}
- Lighting: {analysis.lighting_description}
- Background: {analysis.background_description}

Character name: {character_name}

{caption_prompt}

Generate ONLY the caption, no explanation.
"""
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": context,
                        "images": [image_to_base64(image_path)],
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    return response.json().get('response', '').strip()
                else:
                    return f"[Caption generation failed: {response.status_code}]"
                    
        except Exception as e:
            logger.warning(f"Caption generation error: {e}")
            return f"[Caption generation error: {e}]"


class GeminiProvider(LLMProvider):
    """Google Gemini provider for cloud LLM inference."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._file_cache: dict[str, str] = {}  # path -> file URI
    
    async def check_available(self) -> bool:
        """Check if Gemini API is configured and accessible."""
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models/{self.model}",
                    params={"key": self.api_key}
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Gemini not available: {e}")
            return False
    
    async def enrich_image(
        self,
        image_path: str,
        reference_images: list[str],
        character_name: str,
        embedding_scores: dict,
        prompts: dict,
        caption_formats: list[str]
    ) -> EnrichmentResult:
        """Analyze image using Gemini vision model."""
        result = EnrichmentResult(provider="gemini", model=self.model, generated_at=datetime.now())
        
        try:
            # Build the prompt
            system_prompt = prompts.get('enrichment', {}).get('system', '').format(
                character_name=character_name
            )
            analysis_prompt = prompts.get('enrichment', {}).get('analysis', '').format(
                character_name=character_name,
                face_similarity=embedding_scores.get('face_similarity', 'N/A'),
                shot_type=embedding_scores.get('shot_type', 'unknown')
            )
            
            # Build parts with images inline (base64)
            parts = [{"text": system_prompt}]
            
            # Add reference images (limit to 3)
            for i, ref_path in enumerate(reference_images[:3]):
                if os.path.exists(ref_path):
                    parts.append({"text": f"\n\nReference image {i+1}:"})
                    parts.append({
                        "inline_data": {
                            "mime_type": get_image_mime_type(ref_path),
                            "data": image_to_base64(ref_path)
                        }
                    })
            
            # Add target image
            if os.path.exists(image_path):
                parts.append({"text": "\n\nTarget image to analyze:"})
                parts.append({
                    "inline_data": {
                        "mime_type": get_image_mime_type(image_path),
                        "data": image_to_base64(image_path)
                    }
                })
            else:
                result.error = f"Image not found: {image_path}"
                return result
            
            parts.append({"text": f"\n\n{analysis_prompt}"})
            
            # Call Gemini API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "responseMimeType": "application/json"
                        }
                    }
                )
                
                if response.status_code != 200:
                    result.error = f"Gemini API error: {response.status_code} - {response.text}"
                    return result
                
                response_data = response.json()
                
            # Extract text from response
            candidates = response_data.get('candidates', [])
            if not candidates:
                result.error = "No response from Gemini"
                return result
            
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            response_text = parts[0].get('text', '') if parts else ''
            
            # Parse JSON response
            try:
                # Clean up response text (remove markdown code blocks if present)
                clean_text = response_text.strip()
                if clean_text.startswith('```'):
                    clean_text = clean_text.split('\n', 1)[1]  # Remove first line
                    clean_text = clean_text.rsplit('```', 1)[0]  # Remove last ```
                
                analysis = json.loads(clean_text)
                result.shot_type = analysis.get('shot_type', 'unknown')
                result.pose_description = analysis.get('pose_description', '')
                result.expression = analysis.get('expression', '')
                result.clothing_description = analysis.get('clothing_description', '')
                result.lighting_description = analysis.get('lighting_description', '')
                result.background_description = analysis.get('background_description', '')
                result.full_description = analysis.get('full_description', '')
                result.quality_notes = analysis.get('quality_notes')
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini JSON response: {e}")
                result.full_description = response_text
            
            # Generate captions for each format
            for format_name in caption_formats:
                caption_prompt = prompts.get('captions', {}).get(format_name)
                if caption_prompt:
                    caption = await self._generate_caption(
                        image_path, character_name, result, caption_prompt
                    )
                    result.captions[format_name] = caption
            
        except Exception as e:
            logger.exception(f"Gemini enrichment error: {e}")
            result.error = str(e)
        
        return result
    
    async def _generate_caption(
        self,
        image_path: str,
        character_name: str,
        analysis: EnrichmentResult,
        caption_prompt: str
    ) -> str:
        """Generate a caption for a specific format."""
        try:
            context = f"""
Based on this analysis of the image:
- Shot type: {analysis.shot_type}
- Pose: {analysis.pose_description}
- Expression: {analysis.expression}
- Clothing: {analysis.clothing_description}
- Lighting: {analysis.lighting_description}
- Background: {analysis.background_description}

Character name: {character_name}

{caption_prompt}

Generate ONLY the caption, no explanation.
"""
            parts = [
                {"text": context},
                {
                    "inline_data": {
                        "mime_type": get_image_mime_type(image_path),
                        "data": image_to_base64(image_path)
                    }
                }
            ]
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json={"contents": [{"parts": parts}]}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get('candidates', [])
                    if candidates:
                        content = candidates[0].get('content', {})
                        parts = content.get('parts', [])
                        if parts:
                            return parts[0].get('text', '').strip()
                
                return f"[Caption generation failed: {response.status_code}]"
                    
        except Exception as e:
            logger.warning(f"Caption generation error: {e}")
            return f"[Caption generation error: {e}]"


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider based on settings."""
    settings = load_settings()
    llm_settings = settings.get('llm', {})
    provider = llm_settings.get('provider', 'ollama')
    
    if provider == 'gemini':
        gemini_config = llm_settings.get('gemini', {})
        return GeminiProvider(
            api_key=gemini_config.get('api_key', ''),
            model=gemini_config.get('model', 'gemini-2.0-flash-exp')
        )
    else:
        ollama_config = llm_settings.get('ollama', {})
        return OllamaProvider(
            base_url=ollama_config.get('base_url', 'http://localhost:11434'),
            model=ollama_config.get('model', 'llava:13b')
        )


async def check_llm_status() -> dict:
    """Check status of all configured LLM providers."""
    settings = load_settings()
    llm_settings = settings.get('llm', {})
    
    ollama_config = llm_settings.get('ollama', {})
    ollama = OllamaProvider(
        base_url=ollama_config.get('base_url', 'http://localhost:11434'),
        model=ollama_config.get('model', 'llava:13b')
    )
    
    gemini_config = llm_settings.get('gemini', {})
    gemini = GeminiProvider(
        api_key=gemini_config.get('api_key', ''),
        model=gemini_config.get('model', 'gemini-2.0-flash-exp')
    )
    
    ollama_available = await ollama.check_available()
    gemini_available = await gemini.check_available()
    
    return {
        "active_provider": llm_settings.get('provider', 'ollama'),
        "providers": {
            "ollama": {
                "available": ollama_available,
                "base_url": ollama_config.get('base_url', 'http://localhost:11434'),
                "model": ollama_config.get('model', 'llava:13b')
            },
            "gemini": {
                "available": gemini_available,
                "configured": bool(gemini_config.get('api_key')),
                "model": gemini_config.get('model', 'gemini-2.0-flash-exp')
            }
        }
    }
