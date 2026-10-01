
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping chunks.

    V1 uses a deterministic character-based chunker,
    but prefers whitespace boundaries so words are not
    unnecessarily split.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    text_length = len(text)

    start = 0

    while start < text_length:
        target_end = min(start + chunk_size, text_length)

        # If this is not the final chunk, try to move the
        # boundary backward to the nearest whitespace.
        end = target_end

        if target_end < text_length:
            whitespace_position = text.rfind(" ", start, target_end)

            if whitespace_position > start:
                end = whitespace_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Final chunk.
        if end >= text_length:
            break

        # Move backward for overlap.
        next_start = end - chunk_overlap

        # Avoid getting stuck.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
