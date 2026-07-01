"""Testes unitários para app.schemas.clip e app.schemas.transcript."""

import pytest
from pydantic import ValidationError

from app.schemas.clip import ClipDetectionResult, ViralClip
from app.schemas.transcript import Transcript, TranscriptSegment

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://youtube.com/watch?v={VIDEO_ID}"


def make_clip(**kwargs) -> ViralClip:
    defaults = {
        "title": "Momento viral incrivel no podcast",
        "start_time": 100.0,
        "end_time": 170.0,
        "justification": "Este momento tem alto potencial viral pela emocao gerada.",
        "viral_score": 8.5,
    }
    return ViralClip(**{**defaults, **kwargs})


class TestViralClip:
    def test_clip_valido(self):
        clip = make_clip()
        assert clip.title == "Momento viral incrivel no podcast"
        assert clip.viral_score == 8.5

    def test_duracao_calculada_corretamente(self):
        clip = make_clip(start_time=50.0, end_time=110.0)
        assert clip.duration == 60.0

    def test_end_antes_de_start_levanta_erro(self):
        with pytest.raises(ValidationError):
            make_clip(start_time=100.0, end_time=50.0)

    def test_end_igual_a_start_levanta_erro(self):
        with pytest.raises(ValidationError):
            make_clip(start_time=100.0, end_time=100.0)

    def test_score_acima_do_maximo_levanta_erro(self):
        with pytest.raises(ValidationError):
            make_clip(viral_score=10.1)

    def test_score_abaixo_do_minimo_levanta_erro(self):
        with pytest.raises(ValidationError):
            make_clip(viral_score=-0.1)

    def test_titulo_com_espacos_e_removido(self):
        clip = make_clip(title="  Titulo com espacos  ")
        assert clip.title == "Titulo com espacos"

    def test_formatted_duration_apenas_segundos(self):
        clip = make_clip(start_time=0.0, end_time=45.0)
        assert clip.formatted_duration() == "45s"

    def test_formatted_duration_com_minutos(self):
        clip = make_clip(start_time=0.0, end_time=90.0)
        assert clip.formatted_duration() == "1m 30s"


class TestClipDetectionResult:
    def test_sorted_by_score_ordem_decrescente(self):
        clips = [
            make_clip(viral_score=5.0, start_time=100.0, end_time=170.0),
            make_clip(viral_score=9.0, start_time=200.0, end_time=270.0),
            make_clip(viral_score=7.0, start_time=300.0, end_time=370.0),
        ]
        result = ClipDetectionResult(video_id=VIDEO_ID, url=URL, clips=clips)
        scores = [c.viral_score for c in result.sorted_by_score()]
        assert scores == [9.0, 7.0, 5.0]

    def test_total_clips_found(self):
        clips = [make_clip(), make_clip(start_time=200.0, end_time=270.0)]
        result = ClipDetectionResult(video_id=VIDEO_ID, url=URL, clips=clips)
        assert result.total_clips_found == 2

    def test_sem_clips_retorna_zero(self):
        result = ClipDetectionResult(video_id=VIDEO_ID, url=URL)
        assert result.total_clips_found == 0

    def test_summary_sem_clips(self):
        result = ClipDetectionResult(video_id=VIDEO_ID, url=URL)
        assert "No viral clips" in result.summary()

    def test_summary_com_clips(self):
        result = ClipDetectionResult(video_id=VIDEO_ID, url=URL, clips=[make_clip()])
        summary = result.summary()
        assert "1" in summary
        assert "Momento viral incrivel no podcast" in summary


class TestTranscriptSegment:
    def test_segmento_valido(self):
        seg = TranscriptSegment(start=0.0, end=5.0, text="Ola, bem-vindos ao podcast.")
        assert seg.text == "Ola, bem-vindos ao podcast."

    def test_end_antes_de_start_levanta_erro(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=10.0, end=5.0, text="Texto invalido")

    def test_texto_e_removido_dos_espacos(self):
        seg = TranscriptSegment(start=0.0, end=1.0, text="  texto com espacos  ")
        assert seg.text == "texto com espacos"

    def test_texto_vazio_levanta_erro(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=0.0, end=1.0, text="")


class TestTranscript:
    def _make_transcript(self) -> Transcript:
        segments = [
            TranscriptSegment(start=0.0, end=5.0, text="Segmento um"),
            TranscriptSegment(start=5.0, end=10.0, text="Segmento dois"),
        ]
        return Transcript(
            video_id=VIDEO_ID,
            url=URL,
            language="pt",
            duration=10.0,
            segments=segments,
        )

    def test_segment_count(self):
        transcript = self._make_transcript()
        assert transcript.segment_count == 2

    def test_to_plain_text(self):
        transcript = self._make_transcript()
        assert transcript.to_plain_text() == "Segmento um Segmento dois"

    def test_duracao_invalida_levanta_erro(self):
        with pytest.raises(ValidationError):
            Transcript(video_id=VIDEO_ID, url=URL, language="pt", duration=0.0)
