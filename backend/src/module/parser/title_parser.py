import logging

from module.conf import settings
from module.models import Bangumi
from module.models.bangumi import Episode
from module.parser.analyser import (
    OpenAIParser,
    gather_context,
    mikan_parser,
    raw_parser,
    tmdb_parser,
    torrent_parser,
)

logger = logging.getLogger(__name__)


class TitleParser:
    def __init__(self):
        pass

    @staticmethod
    def torrent_parser(
        torrent_path: str,
        torrent_name: str | None = None,
        season: int | None = None,
        file_type: str = "media",
    ):
        try:
            return torrent_parser(torrent_path, torrent_name, season, file_type)
        except Exception as e:
            logger.warning(f"Cannot parse {torrent_path} with error {e}")

    @staticmethod
    async def tmdb_parser(title: str, season: int, language: str):
        tmdb_info = await tmdb_parser(title, language)
        if tmdb_info:
            logger.debug("TMDB Matched, official title is %s", tmdb_info.title)
            tmdb_season = tmdb_info.last_season if tmdb_info.last_season else season
            return tmdb_info.title, tmdb_season, tmdb_info.year, tmdb_info.poster_link
        else:
            logger.warning(f"Cannot match {title} in TMDB. Use raw title instead.")
            logger.warning("Please change bangumi info manually.")
            return title, season, None, None

    @staticmethod
    async def tmdb_poster_parser(bangumi: Bangumi):
        tmdb_info = await tmdb_parser(
            bangumi.official_title, settings.rss_parser.language
        )
        if tmdb_info:
            logger.debug("TMDB Matched, official title is %s", tmdb_info.title)
            bangumi.poster_link = tmdb_info.poster_link
        else:
            logger.warning(
                f"Cannot match {bangumi.official_title} in TMDB. Use raw title instead."
            )
            logger.warning("Please change bangumi info manually.")

    @staticmethod
    def _llm_parse(raw: str, language: str, context: dict | None = None) -> Episode | None:
        """Call LLM to parse raw title into an Episode.

        Returns None if LLM is disabled or parsing fails.
        """
        if not settings.experimental_openai.enable:
            return None
        if not settings.experimental_openai.api_key:
            logger.debug("LLM api_key is empty, skipping LLM parse")
            return None

        kwargs = settings.experimental_openai.dict(
            exclude={"enable", "parser_strategy"}
        )
        try:
            gpt = OpenAIParser(**kwargs)
            episode_dict = gpt.parse(
                raw, asdict=True, context=context, language=language
            )
            if episode_dict is None:
                return None
            return Episode(**episode_dict)
        except Exception as e:
            logger.warning("LLM parse failed: %s", e)
            return None

    @staticmethod
    def _build_bangumi(episode: Episode, language: str) -> Bangumi | None:
        """Build a Bangumi model from an Episode."""
        titles = {
            "zh": episode.title_zh,
            "en": episode.title_en,
            "jp": episode.title_jp,
        }
        title_raw = episode.title_en or episode.title_zh or episode.title_jp
        if not title_raw:
            logger.warning("Cannot extract title_raw from episode, skipping")
            return None

        if titles[language]:
            official_title = titles[language]
        elif titles["zh"]:
            official_title = titles["zh"]
        elif titles["en"]:
            official_title = titles["en"]
        elif titles["jp"]:
            official_title = titles["jp"]
        else:
            official_title = title_raw

        return Bangumi(
            official_title=official_title,
            title_raw=title_raw,
            season=episode.season,
            season_raw=episode.season_raw,
            group_name=episode.group,
            dpi=episode.resolution,
            source=episode.source,
            subtitle=episode.sub,
            eps_collect=False if episode.episode > 1 else True,
            offset=0,
            filter=",".join(settings.rss_parser.filter),
        )

    @staticmethod
    def raw_parser(raw: str) -> Bangumi | None:
        """Parse a raw torrent title into a Bangumi model.

        Strategy is controlled by ``experimental_openai.parser_strategy``:

        - ``"replace"``: Skip regex entirely, use LLM for every title.
        - ``"fallback"``: Try regex first; if it fails, fall back to LLM.
        """
        language = settings.rss_parser.language
        strategy = settings.experimental_openai.parser_strategy
        try:
            if strategy == "replace":
                # --- Replace mode: LLM for everything ---
                if settings.experimental_openai.enable:
                    episode = TitleParser._llm_parse(raw, language)
                    if episode is None:
                        return None
                else:
                    episode = raw_parser(raw)
                    if episode is None:
                        return None
            else:
                # --- Fallback mode: regex first, LLM on failure ---
                episode = raw_parser(raw)
                if episode is None:
                    logger.warning(
                        "Regex parse failed for '%s', falling back to LLM", raw
                    )
                    ctx = gather_context(raw, language)
                    episode = TitleParser._llm_parse(raw, language, context=ctx)
                    if episode is None:
                        return None
                    logger.info("LLM fallback succeeded for '%s'", raw)

            logger.debug("RAW:%s >> %s", raw, episode.title_en or episode.title_zh)
            return TitleParser._build_bangumi(episode, language)

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Cannot parse '{raw}': {type(e).__name__}: {e}")
            return None

    @staticmethod
    async def mikan_parser(homepage: str) -> tuple[str, str]:
        return await mikan_parser(homepage)
