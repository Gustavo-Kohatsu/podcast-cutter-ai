"""Testes unitários para app.utils.validators."""

import pytest

from app.utils.validators import extract_video_id, validate_youtube_url

VIDEO_ID = "dQw4w9WgXcQ"


class TestExtractVideoId:
    def test_url_watch_padrao(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_url_curta_youtu_be(self):
        assert extract_video_id(f"https://youtu.be/{VIDEO_ID}") == VIDEO_ID

    def test_url_live(self):
        assert extract_video_id(f"https://www.youtube.com/live/{VIDEO_ID}") == VIDEO_ID

    def test_url_shorts(self):
        assert extract_video_id(f"https://www.youtube.com/shorts/{VIDEO_ID}") == VIDEO_ID

    def test_url_mobile(self):
        assert extract_video_id(f"https://m.youtube.com/watch?v={VIDEO_ID}") == VIDEO_ID

    def test_url_com_playlist(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLxxx&index=3"
        assert extract_video_id(url) == VIDEO_ID

    def test_url_sem_video_id_levanta_erro(self):
        with pytest.raises(ValueError):
            extract_video_id("https://www.youtube.com/channel/UC123456789012")

    def test_url_invalida_levanta_erro(self):
        with pytest.raises(ValueError):
            extract_video_id("https://www.youtube.com/feed/subscriptions")


class TestValidateYoutubeUrl:
    def test_url_valida_nao_levanta(self):
        validate_youtube_url(f"https://youtu.be/{VIDEO_ID}")

    def test_url_watch_valida_nao_levanta(self):
        validate_youtube_url(f"https://www.youtube.com/watch?v={VIDEO_ID}")

    def test_dominio_nao_youtube_levanta_erro(self):
        with pytest.raises(ValueError, match="Not a YouTube URL"):
            validate_youtube_url("https://vimeo.com/123456789ab")

    def test_url_vazia_levanta_erro(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_youtube_url("")

    def test_url_apenas_espacos_levanta_erro(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_youtube_url("   ")

    def test_esquema_ftp_levanta_erro(self):
        with pytest.raises(ValueError, match="http or https"):
            validate_youtube_url(f"ftp://www.youtube.com/watch?v={VIDEO_ID}")
