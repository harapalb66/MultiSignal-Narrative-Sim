#!/usr/bin/env python3
"""
Compare narrative structures using LLM reasoning.

Takes extracted structures and uses LLM to determine which story
is more similar to the anchor based on structural similarity.
"""
from core.typing import Optional
from core.pydantic import BaseModel
from core.openai import OpenAI
from core.narrative_structure_extractor import NarrativeStructure


class ComparisonResult(BaseModel):
    """Result of comparing two stories to an anchor."""
    answer: str  # "A" or "B"
    reasoning: str
    confidence: Optional[str] = None  # "high", "medium", "low"


class StructureComparator:
    """Compare narrative structures using LLM reasoning."""

    COMPARISON_PROMPT = """Compare these narrative structures and determine which story (A or B) is more similar to the ANCHOR.

ANCHOR STORY:
- Events: {anchor_events}
- Agents: {anchor_agents}
- Agent-Action-Goal Triples: {anchor_triples}
- Causal Links: {anchor_causal}
- Themes: {anchor_themes}
- Setting: {anchor_setting}
- Genre: {anchor_genre}

STORY A:
- Events: {a_events}
- Agents: {a_agents}
- Agent-Action-Goal Triples: {a_triples}
- Causal Links: {a_causal}
- Themes: {a_themes}
- Setting: {a_setting}
- Genre: {a_genre}

STORY B:
- Events: {b_events}
- Agents: {b_agents}
- Agent-Action-Goal Triples: {b_triples}
- Causal Links: {b_causal}
- Themes: {b_themes}
- Setting: {b_setting}
- Genre: {b_genre}

Analyze the STRUCTURAL similarity (not surface-level details):

1. EVENT SEQUENCE: Which story has more similar plot progression?
2. AGENT ROLES: Which has more similar character dynamics?
3. CAUSAL STRUCTURE: Which has more similar cause-effect patterns?
4. THEMATIC OVERLAP: Which shares more themes with the anchor?
5. GENRE/SETTING MATCH: Which is in a more similar genre/setting?

Based on overall NARRATIVE STRUCTURE similarity, which story (A or B) is MORE SIMILAR to the ANCHOR?

Return JSON with:
- "answer": "A" or "B"
- "reasoning": Brief explanation (2-3 sentences)
- "confidence": "high", "medium", or "low"
"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize comparator.

        Args:
            model: OpenAI model for comparison reasoning
        """
        self.client = OpenAI()
        self.model = model

    def _format_triples(self, triples) -> str:
        """Format triples for prompt."""
        if not triples:
            return "None"
        formatted = []
        for t in triples:
            if t.goal:
                formatted.append(f"({t.agent}, {t.action}, {t.goal})")
            else:
                formatted.append(f"({t.agent}, {t.action})")
        return "; ".join(formatted)

    def _format_causal(self, links) -> str:
        """Format causal links for prompt."""
        if not links:
            return "None"
        return "; ".join([f"{l.cause} → {l.effect}" for l in links])

    def compare(
        self,
        anchor: NarrativeStructure,
        story_a: NarrativeStructure,
        story_b: NarrativeStructure
    ) -> ComparisonResult:
        """
        Compare two stories to an anchor and determine which is more similar.

        Args:
            anchor: The reference narrative structure
            story_a: First candidate structure
            story_b: Second candidate structure

        Returns:
            ComparisonResult with answer ("A" or "B"), reasoning, and confidence
        """
        prompt = self.COMPARISON_PROMPT.format(
            # Anchor
            anchor_events=", ".join(anchor.events),
            anchor_agents=", ".join(anchor.agents),
            anchor_triples=self._format_triples(anchor.triples),
            anchor_causal=self._format_causal(anchor.causal_links),
            anchor_themes=", ".join(anchor.themes),
            anchor_setting=anchor.setting,
            anchor_genre=anchor.genre,
            # Story A
            a_events=", ".join(story_a.events),
            a_agents=", ".join(story_a.agents),
            a_triples=self._format_triples(story_a.triples),
            a_causal=self._format_causal(story_a.causal_links),
            a_themes=", ".join(story_a.themes),
            a_setting=story_a.setting,
            a_genre=story_a.genre,
            # Story B
            b_events=", ".join(story_b.events),
            b_agents=", ".join(story_b.agents),
            b_triples=self._format_triples(story_b.triples),
            b_causal=self._format_causal(story_b.causal_links),
            b_themes=", ".join(story_b.themes),
            b_setting=story_b.setting,
            b_genre=story_b.genre,
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at comparing narrative structures. Analyze the structural similarities and differences, then make a decision."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ComparisonResult,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            print(f"Error in comparison: {e}")
            # Default to A on error
            return ComparisonResult(
                answer="A",
                reasoning=f"Error during comparison: {e}",
                confidence="low"
            )

    def compare_texts(
        self,
        anchor_text: str,
        text_a: str,
        text_b: str,
        extractor
    ) -> ComparisonResult:
        """
        Compare texts directly (extract structures then compare).

        Args:
            anchor_text: The anchor narrative text
            text_a: First candidate text
            text_b: Second candidate text
            extractor: NarrativeStructureExtractor instance

        Returns:
            ComparisonResult
        """
        anchor_struct = extractor.extract(anchor_text)
        a_struct = extractor.extract(text_a)
        b_struct = extractor.extract(text_b)

        return self.compare(anchor_struct, a_struct, b_struct)


def demo_comparison():
    """Demonstrate structure comparison."""
    from core.narrative_structure_extractor import NarrativeStructureExtractor

    # Sample texts
    anchor = """A wealthy widower locks up his two grown-up children, afraid that they will go mad, as did his wife. He then invites a doctor of dubious reputation to supervise their mental health and discover the causes of their apparent madness. Meanwhile, in the vicinity of the mansion, murders are happening in the local village."""

    text_a = """Barbara is married to the distinguished professor of medicine Georg Bertram who once saved her father's life. When they have a mentally handicapped child together his clinical coldness comes to the fore and he wants to commit euthanasia on the child. She stops him and takes the child away to Brittany in the hope that a change of location will help."""

    text_b = """Stefano arrives in a village where he has been employed to restore a fresco depicting what appears to be a killing. While taking up residence in the house previously owned by two sisters of a deceased painter, Stefano begins a romance with a local girl. He learns that the artist, assisted by his insane sisters, had been luring people to his house, murdering them."""

    print("=" * 70)
    print("STRUCTURE COMPARISON DEMO")
    print("=" * 70)

    extractor = NarrativeStructureExtractor(
        model="gpt-4o-mini",
        cache_dir="/opt/semeval_narative/structure_cache"
    )
    comparator = StructureComparator(model="gpt-4o-mini")

    print("\nExtracting structures...")
    anchor_struct = extractor.extract(anchor)
    a_struct = extractor.extract(text_a)
    b_struct = extractor.extract(text_b)

    print("\nComparing structures...")
    result = comparator.compare(anchor_struct, a_struct, b_struct)

    print(f"\n{'=' * 70}")
    print(f"RESULT")
    print(f"{'=' * 70}")
    print(f"\n  Answer: Story {result.answer} is more similar to the anchor")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reasoning: {result.reasoning}")

    print(f"\n  Ground truth: B (this is Example 4 from core.dev set)")
    print(f"  Correct: {'Yes' if result.answer == 'B' else 'No'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_comparison()
