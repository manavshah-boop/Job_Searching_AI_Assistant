from theirstack import get_or_discover_slugs


def test_get_or_discover_slugs_uses_profile_scoped_cache(monkeypatch):
    calls = {"load": [], "save": []}

    def fake_load_discovered_slugs(*, ats, profile=None):
        calls["load"].append((ats, profile))
        return ["cached-gh"] if ats == "greenhouse" else ["cached-lv"]

    def fake_save_discovered_slug(slug, company_name, *, ats="greenhouse", profile=None):
        calls["save"].append((slug, company_name, ats, profile))

    monkeypatch.setattr("theirstack.load_discovered_slugs", fake_load_discovered_slugs)
    monkeypatch.setattr("theirstack.save_discovered_slug", fake_save_discovered_slug)
    monkeypatch.setattr(
        "theirstack.fetch_companies",
        lambda config: [{"name": "Acme", "domain": "acme.com"}],
    )
    monkeypatch.setattr("theirstack.resolve_greenhouse_slug", lambda company: "acme-gh")
    monkeypatch.setattr("theirstack.resolve_lever_slug", lambda company: "acme-lv")
    monkeypatch.setenv("THEIRSTACK_API_KEY", "test-key")

    config = {
        "sources": {
            "greenhouse": {"companies": ["priority-gh"]},
            "lever": {"companies": ["priority-lv"]},
        }
    }

    result = get_or_discover_slugs(config, profile="default")

    assert result == {
        "greenhouse": ["priority-gh", "cached-gh", "acme-gh"],
        "lever": ["priority-lv", "cached-lv", "acme-lv"],
    }
    assert calls["load"] == [
        ("greenhouse", "default"),
        ("lever", "default"),
    ]
    assert calls["save"] == [
        ("acme-gh", "Acme", "greenhouse", "default"),
        ("acme-lv", "Acme", "lever", "default"),
    ]
