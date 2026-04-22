from theirstack import get_or_discover_slugs


def _make_config(gh=None, lv=None, ashby=None, workable=None):
    return {
        "sources": {
            "greenhouse": {"companies": gh or []},
            "lever":      {"companies": lv or []},
            "ashby":      {"companies": ashby or []},
            "workable":   {"companies": workable or []},
        }
    }


def test_returns_all_four_ats_keys():
    config = _make_config()

    def fake_load(*, ats, profile=None):
        return []

    import theirstack
    orig = theirstack.load_discovered_slugs
    theirstack.load_discovered_slugs = fake_load
    try:
        result = get_or_discover_slugs(config)
    finally:
        theirstack.load_discovered_slugs = orig

    assert set(result.keys()) == {"greenhouse", "lever", "ashby", "workable"}


def test_priority_list_comes_first(monkeypatch):
    """Priority slugs from config always appear before cached slugs."""
    def fake_load(*, ats, profile=None):
        return {"greenhouse": ["cached-gh"], "lever": ["cached-lv"],
                "ashby": ["cached-ash"], "workable": ["cached-wl"]}.get(ats, [])

    monkeypatch.setattr("theirstack.load_discovered_slugs", fake_load)

    config = _make_config(
        gh=["priority-gh"],
        lv=["priority-lv"],
        ashby=["priority-ash"],
        workable=["priority-wl"],
    )
    result = get_or_discover_slugs(config)

    assert result["greenhouse"] == ["priority-gh", "cached-gh"]
    assert result["lever"]      == ["priority-lv", "cached-lv"]
    assert result["ashby"]      == ["priority-ash", "cached-ash"]
    assert result["workable"]   == ["priority-wl", "cached-wl"]


def test_cached_slug_not_duplicated_if_in_priority(monkeypatch):
    """Slugs already in priority list are not duplicated from the cache."""
    def fake_load(*, ats, profile=None):
        return ["shared-slug", "extra-cached"]

    monkeypatch.setattr("theirstack.load_discovered_slugs", fake_load)

    config = _make_config(gh=["shared-slug"])
    result = get_or_discover_slugs(config)

    assert result["greenhouse"] == ["shared-slug", "extra-cached"]
    assert result["greenhouse"].count("shared-slug") == 1


def test_profile_scoped_cache(monkeypatch):
    """Cache loads are called with the correct profile argument."""
    calls = []

    def fake_load(*, ats, profile=None):
        calls.append((ats, profile))
        return []

    monkeypatch.setattr("theirstack.load_discovered_slugs", fake_load)

    config = _make_config()
    get_or_discover_slugs(config, profile="manav")

    loaded_ats = {ats for ats, _ in calls}
    assert loaded_ats == {"greenhouse", "lever", "ashby", "workable"}
    assert all(profile == "manav" for _, profile in calls)


def test_empty_config_returns_only_cache(monkeypatch):
    """With no priority companies, result is the DB cache only."""
    def fake_load(*, ats, profile=None):
        return ["cache-only-slug"]

    monkeypatch.setattr("theirstack.load_discovered_slugs", fake_load)

    config = _make_config()
    result = get_or_discover_slugs(config)

    assert result["greenhouse"] == ["cache-only-slug"]
    assert result["ashby"]      == ["cache-only-slug"]


def test_no_external_calls_made(monkeypatch):
    """get_or_discover_slugs must not make any HTTP requests."""
    import httpx

    def explode(*args, **kwargs):
        raise AssertionError("HTTP call made — should not happen")

    monkeypatch.setattr(httpx, "get",  explode)
    monkeypatch.setattr(httpx, "post", explode)
    monkeypatch.setattr("theirstack.load_discovered_slugs", lambda *, ats, profile=None: [])

    config = _make_config(gh=["stripe"], lv=["stripe"], ashby=["linear"], workable=["acme"])
    result = get_or_discover_slugs(config)

    assert result["greenhouse"] == ["stripe"]
    assert result["ashby"]      == ["linear"]
