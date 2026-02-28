#!/usr/bin/env python3
"""
POS-based chunking strategy using verbs and relational structures.

Instead of fixed sentence windows, chunk around:
- Verbs (actions/events - the narrative dynamics)
- Relational POS (conjunctions, prepositions - connections)

This creates semantically meaningful chunks based on narrative structure.
"""
from core.core import spacy
from core.core import numpy as np
from core.narrative_attractor_dynamics import sent_split


class POSChunker:
    """Create chunks based on verb phrases and relational structures."""

    def __init__(self, model="en_core_web_sm"):
        """Initialize with spaCy model for POS tagging."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading spaCy model: {model}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

    def identify_verb_boundaries(self, text):
        """
        Identify verb-based boundaries in text.

        Strategy:
        1. Find main verbs (VERB, AUX)
        2. Find relational elements (CONJ, SCONJ, ADP)
        3. Create chunks around verb phrases

        Returns:
            List of (start_idx, end_idx, chunk_type) tuples
        """
        doc = self.nlp(text)

        # Identify verbs and their spans
        verb_spans = []
        for sent in doc.sents:
            # Find verb chunks in this sentence
            for token in sent:
                if token.pos_ in ['VERB', 'AUX']:
                    # Get the verb phrase span (verb + its dependents)
                    subtree = list(token.subtree)
                    start = min(t.i for t in subtree)
                    end = max(t.i for t in subtree) + 1
                    verb_spans.append({
                        'start': start,
                        'end': end,
                        'verb': token.text,
                        'lemma': token.lemma_,
                        'sent_start': sent.start,
                        'sent_end': sent.end,
                    })

        return verb_spans, doc

    def create_verb_chunks(self, text, min_tokens=10, max_tokens=50):
        """
        Create chunks based on verb phrases and relational boundaries.

        Args:
            text: Input text
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        verb_spans, doc = self.identify_verb_boundaries(text)

        if len(verb_spans) == 0:
            # No verbs found, return whole text
            return [text]

        # Strategy: Group consecutive verb phrases into chunks
        chunks = []
        current_chunk_start = 0
        current_chunk_tokens = []

        for i, verb_span in enumerate(verb_spans):
            # Add tokens from core.current position to this verb phrase
            chunk_end = verb_span['end']

            # Get tokens in this segment
            segment_tokens = doc[current_chunk_start:chunk_end]
            current_chunk_tokens.extend(segment_tokens)

            # Check if we should end this chunk
            should_end = False

            # Reason 1: Chunk is large enough and we hit a sentence boundary
            if len(current_chunk_tokens) >= min_tokens:
                # Check if next verb is in a different sentence
                if i + 1 < len(verb_spans):
                    next_sent = verb_spans[i + 1]['sent_start']
                    curr_sent = verb_span['sent_start']
                    if next_sent != curr_sent:
                        should_end = True
                else:
                    should_end = True  # Last verb

            # Reason 2: Chunk is getting too large
            if len(current_chunk_tokens) >= max_tokens:
                should_end = True

            # Reason 3: Last verb phrase
            if i == len(verb_spans) - 1:
                should_end = True

            if should_end:
                # Create chunk from core.accumulated tokens
                chunk_text = ' '.join([t.text for t in current_chunk_tokens])
                if chunk_text.strip():
                    chunks.append(chunk_text)

                # Reset for next chunk
                current_chunk_start = chunk_end
                current_chunk_tokens = []

        # If we have remaining tokens, add them
        if current_chunk_start < len(doc):
            remaining = doc[current_chunk_start:]
            if len(remaining) > 0:
                chunk_text = ' '.join([t.text for t in remaining])
                if chunk_text.strip():
                    chunks.append(chunk_text)

        # Fallback: if no chunks created, return whole text
        if len(chunks) == 0:
            chunks = [text]

        return chunks

    def create_sentence_verb_chunks(self, text):
        """
        Simpler strategy: Split by sentences, then by main verbs within sentences.

        This creates chunks where each chunk contains:
        - A main verb and its associated clause
        - Typically more chunks than sentence-based approach

        Returns:
            List of text chunks
        """
        sents = sent_split(text)
        chunks = []

        for sent in sents:
            doc = self.nlp(sent)

            # Find main verbs in this sentence
            main_verbs = [token for token in doc if token.pos_ in ['VERB', 'AUX'] and token.dep_ in ['ROOT', 'aux', 'auxpass']]

            if len(main_verbs) == 0:
                # No verbs, add whole sentence
                chunks.append(sent)
            elif len(main_verbs) == 1:
                # Single verb, add whole sentence
                chunks.append(sent)
            else:
                # Multiple verbs, try to split into clauses
                # For now, use simple heuristic: split at coordinating conjunctions
                clause_chunks = []
                current_clause = []

                for token in doc:
                    current_clause.append(token.text)

                    # Split at coordinating conjunctions connecting main verbs
                    if token.pos_ == 'CCONJ' and token.head.pos_ in ['VERB', 'AUX']:
                        clause_text = ' '.join(current_clause[:-1])  # Exclude the CCONJ
                        if clause_text.strip():
                            clause_chunks.append(clause_text)
                        current_clause = []

                # Add remaining clause
                if current_clause:
                    clause_text = ' '.join(current_clause)
                    if clause_text.strip():
                        clause_chunks.append(clause_text)

                # If splitting worked, use clause chunks; otherwise, use whole sentence
                if len(clause_chunks) > 1:
                    chunks.extend(clause_chunks)
                else:
                    chunks.append(sent)

        return chunks

    def create_verb_event_chunks(self, text, events_per_chunk=2):
        """
        Event-based chunking: Group verb events together.

        Each chunk contains approximately events_per_chunk main verb events.
        This captures narrative progression more naturally.

        Args:
            text: Input text
            events_per_chunk: Number of verb events per chunk

        Returns:
            List of text chunks
        """
        doc = self.nlp(text)

        # Identify main verb events (verbs that are ROOT or main clause verbs)
        events = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ in ['VERB', 'AUX'] and token.dep_ in ['ROOT', 'ccomp', 'xcomp', 'conj']:
                    # Get the span of this event (verb + its core arguments)
                    subtree_tokens = list(token.subtree)
                    start = min(t.i for t in subtree_tokens)
                    end = max(t.i for t in subtree_tokens) + 1

                    events.append({
                        'start': start,
                        'end': end,
                        'verb': token.text,
                        'token': token,
                    })

        if len(events) == 0:
            return [text]

        # Group events into chunks
        chunks = []
        i = 0

        while i < len(events):
            # Take events_per_chunk events
            chunk_events = events[i:i + events_per_chunk]

            # Get span from core.first event start to last event end
            chunk_start = chunk_events[0]['start']
            chunk_end = chunk_events[-1]['end']

            # Extract text
            chunk_tokens = doc[chunk_start:chunk_end]
            chunk_text = ' '.join([t.text for t in chunk_tokens])

            if chunk_text.strip():
                chunks.append(chunk_text)

            i += events_per_chunk

        return chunks


def demo_pos_chunking():
    """Demonstrate POS-based chunking on example text."""
    import pandas as pd

    # Load example 4
    dev_df = pd.read_json('SemEval2026-Task_4-dev-v1/dev_track_a.jsonl', lines=True)
    example = dev_df.iloc[4]

    chunker = POSChunker()

    print("=" * 80)
    print("POS-BASED CHUNKING DEMONSTRATION")
    print("=" * 80)

    for text_name, text in [
        ('ANCHOR', example['anchor_text']),
        ('TEXT A', example['text_a']),
        ('TEXT B', example['text_b'])
    ]:
        print(f"\n{'=' * 80}")
        print(f"{text_name}")
        print("=" * 80)

        # Sentence-based chunks (original)
        sents = sent_split(text)
        print(f"\nSentence-based: {len(sents)} sentences")

        # Sentence-verb chunks
        verb_chunks = chunker.create_sentence_verb_chunks(text)
        print(f"Sentence-verb chunks: {len(verb_chunks)} chunks")
        for i, chunk in enumerate(verb_chunks):
            print(f"  Chunk {i}: {chunk[:80]}..." if len(chunk) > 80 else f"  Chunk {i}: {chunk}")

        # Event-based chunks (2 events per chunk)
        event_chunks = chunker.create_verb_event_chunks(text, events_per_chunk=2)
        print(f"\nEvent-based (2 events/chunk): {len(event_chunks)} chunks")
        for i, chunk in enumerate(event_chunks):
            print(f"  Chunk {i}: {chunk[:80]}..." if len(chunk) > 80 else f"  Chunk {i}: {chunk}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    demo_pos_chunking()
