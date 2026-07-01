"""Testes unitários para app.services.chunker."""

import pytest

from app.schemas.transcript import Transcript, TranscriptSegment
from app.services.chunker import ChunkConfig, TranscriptChunker

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://youtube.com/watch?v={VIDEO_ID}"


def make_transcript(duration: float, n_segments: int) -> Transcript:
    """Cria uma transcrição de teste com segmentos distribuídos uniformemente."""
    step = duration / n_segments
    segments = [
        TranscriptSegment(
            start=round(i * step, 2),
            end=round((i + 1) * step, 2),
            text=f"Segmento {i + 1}",
        )
        for i in range(n_segments)
    ]
    return Transcript(
        video_id=VIDEO_ID,
        url=URL,
        language="pt",
        duration=duration,
        segments=segments,
    )


class TestTranscriptChunker:
    def setup_method(self):
        self.chunker = TranscriptChunker()

    def test_transcricao_sem_segmentos_retorna_lista_vazia(self):
        transcript = Transcript(
            video_id=VIDEO_ID,
            url=URL,
            language="pt",
            duration=100.0,
            segments=[],
        )
        chunks = self.chunker.chunk(transcript, ChunkConfig())
        assert chunks == []

    def test_transcricao_curta_gera_um_chunk(self):
        transcript = make_transcript(duration=300.0, n_segments=10)
        chunks = self.chunker.chunk(transcript, ChunkConfig(chunk_duration_s=900.0))
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_transcricao_longa_gera_multiplos_chunks(self):
        transcript = make_transcript(duration=3600.0, n_segments=100)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=0.0)
        chunks = self.chunker.chunk(transcript, config)
        assert len(chunks) == 4

    def test_segmentos_pertencem_ao_chunk_correto(self):
        transcript = make_transcript(duration=1800.0, n_segments=60)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=0.0)
        chunks = self.chunker.chunk(transcript, config)
        for chunk in chunks:
            for seg in chunk.segments:
                assert chunk.start_time <= seg.start < chunk.end_time

    def test_overlap_estende_janela_do_chunk(self):
        transcript = make_transcript(duration=1800.0, n_segments=60)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=45.0)
        chunks = self.chunker.chunk(transcript, config)
        assert chunks[0].end_time == pytest.approx(945.0)

    def test_ultimo_chunk_nao_ultrapassa_duracao_total(self):
        transcript = make_transcript(duration=1800.0, n_segments=60)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=45.0)
        chunks = self.chunker.chunk(transcript, config)
        assert chunks[-1].end_time <= transcript.duration

    def test_video_id_preservado_nos_chunks(self):
        transcript = make_transcript(duration=300.0, n_segments=10)
        chunks = self.chunker.chunk(transcript, ChunkConfig())
        assert all(c.video_id == VIDEO_ID for c in chunks)

    def test_label_do_chunk(self):
        transcript = make_transcript(duration=1800.0, n_segments=60)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=0.0)
        chunks = self.chunker.chunk(transcript, config)
        assert chunks[0].label == "Chunk 1/2"
        assert chunks[1].label == "Chunk 2/2"

    def test_duracao_do_chunk(self):
        transcript = make_transcript(duration=1800.0, n_segments=60)
        config = ChunkConfig(chunk_duration_s=900.0, overlap_s=0.0)
        chunks = self.chunker.chunk(transcript, config)
        assert chunks[0].duration == pytest.approx(900.0)

    def test_to_timestamped_text_contem_segmentos(self):
        transcript = make_transcript(duration=300.0, n_segments=5)
        chunks = self.chunker.chunk(transcript, ChunkConfig())
        text = chunks[0].to_timestamped_text()
        assert "Segmento 1" in text
        assert "s -" in text
