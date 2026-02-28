#!/usr/bin/env python3
"""
Extract structured narrative representations using LLM.

Converts stories into:
- Event chains
- Agent-action-goal triples
- Causal links
- Themes, setting, genre
"""
from core.core import json
from core.core import hashlib
import os
from core.typing import List, Tuple, Optional
from core.pydantic import BaseModel
from core.openai import OpenAI


# Pydantic models for structured output
class AgentActionGoal(BaseModel):
    """A triple representing who does what for what purpose."""
    agent: str
    action: str
    goal: Optional[str] = None


class CausalLink(BaseModel):
    """A causal relationship between two events."""
    cause: str
    effect: str


class NarrativeStructure(BaseModel):
    """Complete structured representation of a narrative."""
    events: List[str]                    # Chronological event chain
    agents: List[str]                    # Characters/actors
    triples: List[AgentActionGoal]       # Agent-action-goal tuples
    causal_links: List[CausalLink]       # Cause-effect relationships
    themes: List[str]                    # Main themes
    setting: str                         # Where/when
    genre: str                           # Story type


class NarrativeStructureExtractor:
    """Extract structured representations from core.narrative text using LLM."""

    EXTRACTION_PROMPT = """Extract the narrative structure from core.this story synopsis.

STORY:
{story_text}

Extract and return a JSON object with:

1. "events": List of main events in chronological order (3-7 events)
   - Each event should be a short phrase describing what happens
   - Focus on plot-driving actions, not descriptions

2. "agents": List of main characters/actors in the story
   - Include named characters and key roles (e.g., "hero", "villain")

3. "triples": List of agent-action-goal structures
   - Each triple has: "agent" (who), "action" (does what), "goal" (optional: why/for what)
   - Example: {{"agent": "hero", "action": "fights dragon", "goal": "save princess"}}

4. "causal_links": List of cause-effect relationships
   - Each link has: "cause" (event that causes), "effect" (resulting event)
   - Example: {{"cause": "hero arrives", "effect": "villain flees"}}

5. "themes": List of main themes (2-4 themes)
   - Examples: "revenge", "redemption", "love", "survival", "betrayal"

6. "setting": Brief description of where/when the story takes place

7. "genre": The story's genre (e.g., "thriller", "romance", "sci-fi", "drama")

Focus on the NARRATIVE STRUCTURE, not superficial details. Two stories about different characters in different settings can have the same structure (e.g., both are "hero's journey" stories)."""

    def __init__(self, model: str = "gpt-4o-mini", cache_dir: str = None):
        """
        Initialize extractor.

        Args:
            model: OpenAI model to use
            cache_dir: Directory to cache extracted structures (None = no caching)
        """
        self.client = OpenAI()
        self.model = model
        self.cache_dir = cache_dir

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, text: str) -> str:
        """Get cache file path for a text."""
        if not self.cache_dir:
            return None
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.json")

    def _load_from_cache(self, text: str) -> Optional[NarrativeStructure]:
        """Load cached structure if available."""
        cache_path = self._get_cache_path(text)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return NarrativeStructure(**data)
            except Exception:
                return None
        return None

    def _save_to_cache(self, text: str, structure: NarrativeStructure):
        """Save extracted structure to cache."""
        cache_path = self._get_cache_path(text)
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump(structure.model_dump(), f, indent=2)

    def extract(self, story_text: str) -> NarrativeStructure:
        """
        Extract structured representation from core.a story.

        Args:
            story_text: The narrative text to analyze

        Returns:
            NarrativeStructure with events, agents, triples, etc.
        """
        # Check cache first
        cached = self._load_from_cache(story_text)
        if cached:
            return cached

        # Call LLM for extraction
        prompt = self.EXTRACTION_PROMPT.format(story_text=story_text)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing narrative structure. Extract the requested information precisely and return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format=NarrativeStructure,
            )

            structure = response.choices[0].message.parsed

            # Cache the result
            self._save_to_cache(story_text, structure)

            return structure

        except Exception as e:
            print(f"Error extracting structure: {e}")
            # Return a minimal structure on error
            return NarrativeStructure(
                events=["unknown"],
                agents=["unknown"],
                triples=[AgentActionGoal(agent="unknown", action="unknown")],
                causal_links=[],
                themes=["unknown"],
                setting="unknown",
                genre="unknown"
            )

    def extract_batch(self, texts: List[str], show_progress: bool = True) -> List[NarrativeStructure]:
        """
        Extract structures from core.multiple texts.

        Args:
            texts: List of narrative texts
            show_progress: Whether to print progress

        Returns:
            List of NarrativeStructure objects
        """
        results = []
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                print(f"  Extracting {i}/{len(texts)}...")
            results.append(self.extract(text))
        return results


def demo_extraction():
    """Demonstrate extraction on a sample text."""
    sample_text = """A wealthy widower locks up his two grown-up children, afraid that they will go mad, as did his wife. He then invites a doctor of dubious reputation to supervise their mental health and discover the causes of their apparent madness. Meanwhile, in the vicinity of the mansion, murders are happening in the local village."""

    print("=" * 70)
    print("NARRATIVE STRUCTURE EXTRACTION DEMO")
    print("=" * 70)

    print("\nOriginal text:")
    print(f"  {sample_text[:200]}...")

    extractor = NarrativeStructureExtractor(
        model="gpt-4o-mini",
        cache_dir="/opt/semeval_narative/structure_cache"
    )

    print("\nExtracting structure...")
    structure = extractor.extract(sample_text)

    print("\nExtracted Structure:")
    print(f"\n  Events ({len(structure.events)}):")
    for e in structure.events:
        print(f"    - {e}")

    print(f"\n  Agents ({len(structure.agents)}):")
    for a in structure.agents:
        print(f"    - {a}")

    print(f"\n  Agent-Action-Goal Triples ({len(structure.triples)}):")
    for t in structure.triples:
        goal_str = f" → {t.goal}" if t.goal else ""
        print(f"    - {t.agent} | {t.action}{goal_str}")

    print(f"\n  Causal Links ({len(structure.causal_links)}):")
    for c in structure.causal_links:
        print(f"    - {c.cause} → {c.effect}")

    print(f"\n  Themes: {', '.join(structure.themes)}")
    print(f"  Setting: {structure.setting}")
    print(f"  Genre: {structure.genre}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_extraction()
