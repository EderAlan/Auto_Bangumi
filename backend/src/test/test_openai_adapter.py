"""Tests for the DeepSeek adapter changes: dual strategy, cache, retry, context.

These tests verify the new code paths without making real API calls.
"""
import json
from unittest import mock

import pytest

from module.conf import settings
from module.models.bangumi import Episode
from module.parser.analyser.openai import (
    OpenAIParser,
    _cache,
    _cache_get,
    _cache_put,
    _parse_json_response,
    extract_anime_name,
)
from module.parser.title_parser import TitleParser


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJSONResponse:
    def test_valid_json(self):
        assert _parse_json_response('{"a": 1}') == {"a": 1}

    def test_json_with_markdown_wrapper(self):
        result = _parse_json_response('```json\n{"b": 2}\n```')
        assert result == {"b": 2}

    def test_empty_string(self):
        assert _parse_json_response("") is None

    def test_invalid_json_no_braces(self):
        assert _parse_json_response("not json at all") is None

    def test_extracts_first_json_object(self):
        content = 'Prefix text {"key": "value"} suffix'
        assert _parse_json_response(content) == {"key": "value"}


# ---------------------------------------------------------------------------
# extract_anime_name
# ---------------------------------------------------------------------------


class TestExtractAnimeName:
    def test_strips_group_brackets(self):
        name = extract_anime_name(
            "[Sakurato] Kono Subarashii Sekai ni Shukufuku wo! [1080p]"
        )
        assert "Sakurato" not in name
        assert "1080p" not in name

    def test_strips_episode_marker(self):
        name = extract_anime_name("葬送的芙莉莲 第05話 1080p")
        assert "第05話" not in name
        assert "1080p" not in name

    def test_strips_season_marker(self):
        name = extract_anime_name("某番剧 第3季 EP01 HEVC")
        assert "第3季" not in name
        assert "EP01" not in name

    def test_strips_resolution_and_codec(self):
        name = extract_anime_name("Title 1080p HEVC MP4")
        assert "1080p" not in name
        assert "HEVC" not in name
        assert "MP4" not in name


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class TestCache:
    def setup_method(self):
        _cache.clear()

    def test_cache_miss_returns_none(self):
        assert _cache_get("some title") is None

    def test_cache_put_and_get_success(self):
        _cache_put("title A", {"result": "ok"})
        assert _cache_get("title A") == {"result": "ok"}

    def test_cache_put_and_get_failure(self):
        _cache_put("title B", None)
        from module.parser.analyser.openai import _SENTINEL

        assert _cache_get("title B") is _SENTINEL

    def test_cache_reorders_on_hit(self):
        from module.parser.analyser.openai import _SENTINEL

        _cache_put("old", None)
        _cache_put("new", {"data": 1})
        # Access old → moves to end
        assert _cache_get("old") is _SENTINEL
        # Verify order
        keys = list(_cache.keys())
        assert keys[-1] == "old"


# ---------------------------------------------------------------------------
# OpenAIParser._prepare_params — verify json_object mode, not beta.parse
# ---------------------------------------------------------------------------


class TestPrepareParams:
    def setup_method(self):
        self.parser = OpenAIParser(api_key="test-key")

    def test_uses_json_object_response_format(self):
        msgs = [{"role": "user", "content": "hello"}]
        params = self.parser._prepare_params(msgs)
        assert params["response_format"] == {"type": "json_object"}
        assert params["temperature"] == 0.0

    def test_does_not_use_response_format_model(self):
        msgs = [{"role": "user", "content": "hello"}]
        params = self.parser._prepare_params(msgs)
        # The old beta.parse used response_format=Episode
        assert "response_format" not in str(type(params.get("response_format")))
        assert isinstance(params["response_format"], dict)


# ---------------------------------------------------------------------------
# TitleParser._build_bangumi
# ---------------------------------------------------------------------------


class TestBuildBangumi:
    def test_builds_with_all_titles(self):
        ep = Episode(
            title_en="Attack on Titan",
            title_zh="进击的巨人",
            title_jp="進撃の巨人",
            season=3,
            season_raw="S3",
            episode=12,
            sub="CHS",
            group="Lilith-Raws",
            resolution="1080p",
            source="Baha",
        )
        bangumi = TitleParser._build_bangumi(ep, language="zh")
        assert bangumi.official_title == "进击的巨人"
        assert bangumi.title_raw == "Attack on Titan"
        assert bangumi.season == 3
        assert bangumi.group_name == "Lilith-Raws"

    def test_falls_back_to_en_when_zh_missing(self):
        ep = Episode(
            title_en="Attack on Titan",
            title_zh=None,
            title_jp=None,
            season=1,
            season_raw="",
            episode=5,
            sub="",
            group="",
            resolution="",
            source="",
        )
        bangumi = TitleParser._build_bangumi(ep, language="zh")
        assert bangumi.official_title == "Attack on Titan"

    def test_returns_none_when_all_titles_empty(self):
        ep = Episode(
            title_en=None,
            title_zh=None,
            title_jp=None,
            season=1,
            season_raw="",
            episode=1,
            sub="",
            group="",
            resolution="",
            source="",
        )
        assert TitleParser._build_bangumi(ep, language="zh") is None

    def test_eps_collect_true_for_episode_1(self):
        ep = Episode(
            title_en="Test",
            title_zh=None,
            title_jp=None,
            season=1,
            season_raw="",
            episode=1,
            sub="",
            group="",
            resolution="",
            source="",
        )
        bangumi = TitleParser._build_bangumi(ep, language="en")
        assert bangumi.eps_collect is True

    def test_eps_collect_false_for_episode_2(self):
        ep = Episode(
            title_en="Test",
            title_zh=None,
            title_jp=None,
            season=1,
            season_raw="",
            episode=2,
            sub="",
            group="",
            resolution="",
            source="",
        )
        bangumi = TitleParser._build_bangumi(ep, language="en")
        assert bangumi.eps_collect is False


# ---------------------------------------------------------------------------
# TitleParser.raw_parser — dual strategy
# ---------------------------------------------------------------------------


class TestRawParserFallbackStrategy:
    """Verify fallback mode: regex first, LLM only on failure."""

    @pytest.fixture(autouse=True)
    def _manage_settings(self):
        """Save & restore settings to prevent test pollution."""
        saved = (
            settings.experimental_openai.enable,
            settings.experimental_openai.parser_strategy,
            settings.experimental_openai.api_key,
        )
        settings.experimental_openai.enable = True
        settings.experimental_openai.parser_strategy = "fallback"
        settings.experimental_openai.api_key = "test-key"
        yield
        settings.experimental_openai.enable = saved[0]
        settings.experimental_openai.parser_strategy = saved[1]
        settings.experimental_openai.api_key = saved[2]

    def test_regex_succeeds_no_llm_call(self):
        """When regex parses successfully, LLM is never invoked."""
        with mock.patch(
            "module.parser.title_parser.TitleParser._llm_parse"
        ) as mock_llm:
            result = TitleParser.raw_parser(
                "[Lilith-Raws] Otonari no Tenshi-sama - 09 [Baha][WEB-DL][1080p][AVC AAC][CHT][MP4]"
            )
            mock_llm.assert_not_called()
            assert result is not None

    def test_regex_fails_llm_fallback_called(self):
        """When regex fails, LLM fallback is called."""
        with mock.patch(
            "module.parser.title_parser.TitleParser._llm_parse"
        ) as mock_llm:
            ep = Episode(
                title_en="Test Anime",
                title_zh="测试动画",
                title_jp=None,
                season=1,
                season_raw="",
                episode=5,
                sub="",
                group="TestGroup",
                resolution="1080p",
                source="",
            )
            mock_llm.return_value = ep
            # A title that regex cannot parse
            result = TitleParser.raw_parser(
                "[TestGroup] Some Really Weird Format Without Proper Separators"
            )
            mock_llm.assert_called_once()
            assert result is not None
            assert result.official_title == "测试动画"

    def test_both_regex_and_llm_fail_returns_none(self):
        """When both regex and LLM fail, returns None."""
        with mock.patch(
            "module.parser.title_parser.TitleParser._llm_parse"
        ) as mock_llm:
            mock_llm.return_value = None
            result = TitleParser.raw_parser(
                "completely unparseable garbage text !@#$%"
            )
            assert result is None

    def test_fallback_with_llm_disabled_still_uses_regex(self):
        """With LLM disabled in fallback mode, regex-only behavior."""
        settings.experimental_openai.enable = False
        result = TitleParser.raw_parser(
            "[Lilith-Raws] Otonari no Tenshi-sama - 09 [1080p]"
        )
        assert result is not None


class TestRawParserReplaceStrategy:
    """Verify replace mode: LLM replaces regex for all titles."""

    @pytest.fixture(autouse=True)
    def _manage_settings(self):
        """Save & restore settings to prevent test pollution."""
        saved = (
            settings.experimental_openai.enable,
            settings.experimental_openai.parser_strategy,
            settings.experimental_openai.api_key,
        )
        settings.experimental_openai.enable = True
        settings.experimental_openai.parser_strategy = "replace"
        settings.experimental_openai.api_key = "test-key"
        yield
        settings.experimental_openai.enable = saved[0]
        settings.experimental_openai.parser_strategy = saved[1]
        settings.experimental_openai.api_key = saved[2]

    def test_replace_uses_llm_for_valid_title(self):
        """Even perfectly parseable titles go through LLM in replace mode."""
        ep = Episode(
            title_en="Test Anime",
            title_zh="测试",
            title_jp=None,
            season=1,
            season_raw="",
            episode=1,
            sub="",
            group="TestGroup",
            resolution="1080p",
            source="",
        )
        with mock.patch(
            "module.parser.title_parser.TitleParser._llm_parse"
        ) as mock_llm:
            mock_llm.return_value = ep
            result = TitleParser.raw_parser(
                "[TestGroup] Test Anime - 01 [1080p]"
            )
            mock_llm.assert_called_once()
            assert result is not None

    def test_replace_with_llm_disabled_falls_back_to_regex(self):
        """LLM disabled in replace mode → fall through to regex."""
        settings.experimental_openai.enable = False
        result = TitleParser.raw_parser(
            "[Lilith-Raws] Otonari no Tenshi-sama - 09 [1080p]"
        )
        assert result is not None
        assert result.official_title == "Otonari no Tenshi-sama"


class TestLLMParseNoAPIKey:
    """LLM parse should return None when no api_key is configured."""

    def test_no_api_key_returns_none(self):
        saved = (
            settings.experimental_openai.enable,
            settings.experimental_openai.api_key,
        )
        settings.experimental_openai.enable = True
        settings.experimental_openai.api_key = ""
        try:
            result = TitleParser._llm_parse("some title", "zh")
            assert result is None
        finally:
            settings.experimental_openai.enable = saved[0]
            settings.experimental_openai.api_key = saved[1]


# ---------------------------------------------------------------------------
# OpenAIParser.parse — cache integration
# ---------------------------------------------------------------------------


class TestParserCacheIntegration:
    def setup_method(self):
        _cache.clear()
        self.parser = OpenAIParser(api_key="test-key")

    def test_second_call_hits_cache(self):
        expected = {"group": "X", "title_en": "Y", "season": 1, "episode": 5}

        with mock.patch.object(
            self.parser, "_call_with_retry", return_value=expected
        ) as mock_call:
            # First call
            result1 = self.parser.parse("test title")
            assert result1 == expected
            assert mock_call.call_count == 1

            # Second call — should hit cache
            result2 = self.parser.parse("test title")
            assert result2 == expected
            assert mock_call.call_count == 1  # Not called again

    def test_failed_call_is_cached(self):
        with mock.patch.object(
            self.parser, "_call_with_retry", return_value=None
        ) as mock_call:
            # First call — fails
            result1 = self.parser.parse("bad title")
            assert result1 is None
            assert mock_call.call_count == 1

            # Second call — should hit failure cache
            result2 = self.parser.parse("bad title")
            assert result2 is None
            assert mock_call.call_count == 1  # Not called again


# ---------------------------------------------------------------------------
# OpenAIParser — retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    def setup_method(self):
        self.parser = OpenAIParser(api_key="test-key")

    def test_retries_once_on_failure(self):
        msgs = [{"role": "user", "content": "hello"}]
        call_count = 0

        def failing_create(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("test error")

        with mock.patch.object(
            self.parser.client.chat.completions, "create", side_effect=failing_create
        ):
            result = self.parser._call_with_retry(msgs)
            assert result is None
            assert call_count == 2  # 1 initial + 1 retry

    def test_succeeds_on_retry(self):
        msgs = [{"role": "user", "content": "hello"}]
        call_count = 0

        def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("test error")
            resp = mock.MagicMock()
            resp.choices = [mock.MagicMock()]
            resp.choices[0].message.content = '{"result": "ok"}'
            return resp

        with mock.patch.object(
            self.parser.client.chat.completions, "create", side_effect=flaky_create
        ):
            result = self.parser._call_with_retry(msgs)
            assert result == {"result": "ok"}
            assert call_count == 2
