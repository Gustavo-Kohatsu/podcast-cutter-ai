"""Testes unitários para app.services.clip_detector.

Testa apenas os métodos de parsing e expansão de clips, que contêm
a lógica mais crítica do pipeline — sem fazer chamadas reais ao Ollama.
"""

import json
from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.schemas.clip import ViralClip
from app.services.clip_detector import ClipDetector

VIDEO_DURATION = 1000.0


@pytest.fixture
def settings() -> Settings:
    return Settings(
        min_clip_duration=60,
        max_clip_duration=120,
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture
def detector(settings: Settings) -> ClipDetector:
    with patch("app.services.clip_detector.ollama.Client"):
        return ClipDetector(settings)


def clips_json(*clips: dict) -> str:
    """Serializa uma lista de clips no formato JSON esperado pelo LLM."""
    return json.dumps({"clips": list(clips)})


def valid_clip_data(**kwargs) -> dict:
    defaults = {
        "title": "Revelacao surpreendente no podcast ao vivo",
        "start_time": 400.0,
        "end_time": 470.0,
        "justification": "Momento de alta carga emocional com reacao do publico.",
        "viral_score": 8.5,
    }
    return {**defaults, **kwargs}


class TestParseClips:
    def test_clip_valido_e_aceito(self, detector):
        data = clips_json(valid_clip_data())
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert len(clips) == 1
        assert clips[0].viral_score == 8.5

    def test_json_invalido_retorna_lista_vazia(self, detector):
        clips = detector._parse_clips("isso nao e json valido {", VIDEO_DURATION + 5)
        assert clips == []

    def test_titulo_placeholder_descartado(self, detector):
        data = clips_json(valid_clip_data(title="Título real e criativo do momento viral"))
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert clips == []

    def test_titulo_vazio_descartado(self, detector):
        data = clips_json(valid_clip_data(title=""))
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert clips == []

    def test_end_time_acima_da_duracao_descartado(self, detector):
        data = clips_json(valid_clip_data(start_time=100.0, end_time=9999.0))
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert clips == []

    def test_clip_muito_longo_descartado(self, detector):
        data = clips_json(valid_clip_data(start_time=100.0, end_time=500.0))
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert clips == []

    def test_clip_curto_expandido_ate_minimo(self, detector):
        """Clip de 10s deve ser expandido para 60s (min_clip_duration)."""
        data = clips_json(valid_clip_data(start_time=500.0, end_time=510.0))
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        assert len(clips) == 1
        assert clips[0].duration >= 60.0

    def test_viral_score_omitido_usa_padrao(self, detector):
        raw = valid_clip_data()
        del raw["viral_score"]
        clips = detector._parse_clips(clips_json(raw), VIDEO_DURATION + 5)
        assert len(clips) == 1
        assert clips[0].viral_score == 7.0

    def test_campo_justification_maiusculo_normalizado(self, detector):
        raw = valid_clip_data()
        raw["Justification"] = raw.pop("justification")
        clips = detector._parse_clips(clips_json(raw), VIDEO_DURATION + 5)
        assert len(clips) == 1

    def test_multiplos_clips_ordenados_por_score_decrescente(self, detector):
        data = clips_json(
            valid_clip_data(start_time=100.0, end_time=170.0, viral_score=6.0),
            valid_clip_data(start_time=300.0, end_time=370.0, viral_score=9.0),
            valid_clip_data(start_time=500.0, end_time=570.0, viral_score=7.5),
        )
        clips = detector._parse_clips(data, VIDEO_DURATION + 5)
        scores = [c.viral_score for c in clips]
        assert scores == sorted(scores, reverse=True)

    def test_lista_vazia_no_json_retorna_vazio(self, detector):
        clips = detector._parse_clips(json.dumps({"clips": []}), VIDEO_DURATION + 5)
        assert clips == []


class TestExpandClip:
    def _make_clip(self, start: float, end: float) -> ViralClip:
        return ViralClip(
            title="Clip de teste para expansao",
            start_time=start,
            end_time=end,
            justification="Justificativa de teste para validacao do schema.",
            viral_score=8.0,
        )

    def test_clip_curto_expandido_para_minimo(self, detector):
        clip = self._make_clip(500.0, 510.0)
        expanded = detector._expand_clip(clip, video_end=1000.0)
        assert expanded.duration >= 60.0

    def test_centro_do_clip_preservado(self, detector):
        clip = self._make_clip(500.0, 510.0)
        centro_original = (clip.start_time + clip.end_time) / 2
        expanded = detector._expand_clip(clip, video_end=1000.0)
        centro_expandido = (expanded.start_time + expanded.end_time) / 2
        assert abs(centro_expandido - centro_original) < 1.0

    def test_start_nao_fica_negativo_no_inicio_do_video(self, detector):
        clip = self._make_clip(5.0, 10.0)
        expanded = detector._expand_clip(clip, video_end=1000.0)
        assert expanded.start_time >= 0.0
        assert expanded.duration >= 60.0

    def test_end_nao_ultrapassa_fim_do_video(self, detector):
        clip = self._make_clip(990.0, 995.0)
        expanded = detector._expand_clip(clip, video_end=1000.0)
        assert expanded.end_time <= 1000.0

    def test_clip_ja_no_minimo_nao_e_alterado(self, detector):
        clip = self._make_clip(100.0, 160.0)
        result = detector._expand_clip(clip, video_end=1000.0)
        assert result.start_time == clip.start_time
        assert result.end_time == clip.end_time

    def test_clip_maior_que_minimo_nao_e_alterado(self, detector):
        clip = self._make_clip(100.0, 190.0)
        result = detector._expand_clip(clip, video_end=1000.0)
        assert result.start_time == clip.start_time
        assert result.end_time == clip.end_time
