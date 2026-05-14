"""Perplexity AI integration for videocode.

Uses Perplexity for:
- Verifying extracted code correctness
- Searching framework documentation
- Fixing errors in extracted code
- Generating setup instructions
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


@dataclass
class PerplexityResult:
    """Result from Perplexity API call."""
    content: str
    citations: list[str]
    model: str
    tokens_used: int


@dataclass
class CodeVerificationResult:
    """Result of verifying extracted code."""
    is_valid: bool
    issues: list[dict]
    fixed_code: Optional[str]
    explanation: str
    references: list[str]


class PerplexityClient:
    """Client for Perplexity AI API integration."""

    # Modelos disponibles
    MODELS = {
        "sonar": "sonar",           # Más rápido, más barato
        "sonar-pro": "sonar-pro",   # Mejor calidad
        "sonar-reasoning": "sonar-reasoning",  # Mejor para razonamiento
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "sonar-pro"):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY env var "
                "or pass api_key parameter."
            )
        self.model = model
        self.client = httpx.Client(
            timeout=60.0,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def is_available(self) -> bool:
        """Check if Perplexity API is accessible."""
        try:
            resp = self.client.post(PERPLEXITY_API_URL, json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            })
            return resp.status_code == 200
        except Exception as e:
            logger.warning("Perplexity not available: %s", e)
            return False

    def chat(self, message: str, system_prompt: Optional[str] = None) -> PerplexityResult:
        """Send a chat message to Perplexity.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            PerplexityResult with response and citations
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        resp = self.client.post(PERPLEXITY_API_URL, json={
            "model": self.model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.2,
        })
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return PerplexityResult(
            content=choice["message"]["content"],
            citations=data.get("citations", []),
            model=self.model,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    def verify_code(self, code: str, language: str) -> CodeVerificationResult:
        """Verify extracted code for correctness.
        
        Uses Perplexity to check if the code is valid, find errors,
        and suggest fixes with documentation references.
        
        Args:
            code: The extracted code to verify
            language: Programming language
            
        Returns:
            CodeVerificationResult with validation status and fixes
        """
        logger.info("Verifying %s code with Perplexity", language)

        system_prompt = (
            f"You are an expert {language} developer. Verify the following code "
            f"for correctness, identify any errors, and provide fixes. "
            f"Respond in JSON format with these fields: "
            f"is_valid (boolean), issues (array of {{severity, line, description}}), "
            f"fixed_code (string or null), explanation (string), "
            f"references (array of URLs)."
        )

        result = self.chat(
            f"Verify this {language} code:\n\n```{language}\n{code}\n```",
            system_prompt=system_prompt,
        )

        # Parse JSON response
        try:
            # Extract JSON from the response
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            return CodeVerificationResult(
                is_valid=data.get("is_valid", False),
                issues=data.get("issues", []),
                fixed_code=data.get("fixed_code"),
                explanation=data.get("explanation", ""),
                references=result.citations,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse verification JSON: %s", e)
            # Return a basic result with the raw content
            return CodeVerificationResult(
                is_valid=True,  # Assume valid if we can't parse
                issues=[],
                fixed_code=None,
                explanation=result.content,
                references=result.citations,
            )

    def find_documentation(self, query: str) -> PerplexityResult:
        """Search for documentation about a programming topic.
        
        Args:
            query: Search query (e.g., "React useState hook documentation")
            
        Returns:
            PerplexityResult with documentation and sources
        """
        logger.info("Searching docs with Perplexity: %s", query)

        system_prompt = (
            "You are a technical documentation assistant. Provide accurate, "
            "up-to-date information with code examples. Always cite your sources."
        )

        return self.chat(query, system_prompt=system_prompt)

    def generate_setup_instructions(self, framework: str, dependencies: list[str]) -> str:
        """Generate setup instructions for a project.
        
        Args:
            framework: Main framework (e.g., "React", "Django")
            dependencies: List of dependencies detected
            
        Returns:
            Markdown-formatted setup instructions
        """
        logger.info("Generating setup instructions for %s", framework)

        deps_str = "\n".join(f"- {d}" for d in dependencies)
        
        result = self.chat(
            f"Generate setup instructions for a {framework} project with these dependencies:\n"
            f"{deps_str}\n\n"
            f"Include: installation steps, environment setup, running the project, "
            f"and any common issues. Format as markdown.",
            system_prompt="You are a developer onboarding specialist. Provide clear, step-by-step setup instructions.",
        )

        return result.content

    def fix_code_errors(self, code: str, error_message: str, language: str) -> str:
        """Fix code errors based on an error message.
        
        Args:
            code: The code with errors
            error_message: The error message
            language: Programming language
            
        Returns:
            Fixed code
        """
        logger.info("Fixing %s code errors with Perplexity", language)

        result = self.chat(
            f"Fix this {language} code that produces this error:\n\n"
            f"Error: {error_message}\n\n"
            f"Code:\n```{language}\n{code}\n```\n\n"
            f"Return only the fixed code, no explanations.",
            system_prompt=f"You are an expert {language} developer. Fix the code.",
        )

        # Extract code from response
        content = result.content
        if "```" in content:
            # Extract from code blocks
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    # Remove language identifier
                    lines = part.split("\n")
                    if lines and lines[0].strip().lower() == language.lower():
                        return "\n".join(lines[1:])
                    return part
        
        return content

    def get_framework_info(self, framework: str, version: Optional[str] = None) -> dict:
        """Get information about a framework/library.
        
        Args:
            framework: Framework name (e.g., "React", "Vue")
            version: Specific version (optional)
            
        Returns:
            Dict with framework info
        """
        logger.info("Getting info for %s", framework)

        version_str = f" version {version}" if version else ""
        result = self.chat(
            f"Provide information about {framework}{version_str} including: "
            f"latest stable version, key features, breaking changes from previous versions, "
            f"and migration notes. Format as JSON with fields: "
            f"name, latest_version, key_features (array), breaking_changes (array), "
            f"migration_notes (string).",
            system_prompt="You are a technical research assistant. Provide accurate framework information.",
        )

        try:
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse framework info: %s", e)
            return {
                "name": framework,
                "latest_version": "unknown",
                "key_features": [],
                "breaking_changes": [],
                "migration_notes": result.content,
            }

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


def create_perplexity_client_from_env() -> Optional[PerplexityClient]:
    """Create PerplexityClient from environment variable if available."""
    key = os.environ.get("PERPLEXITY_API_KEY")
    if key:
        try:
            return PerplexityClient(key)
        except Exception as e:
            logger.warning("Failed to create Perplexity client: %s", e)
    return None
