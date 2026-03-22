from biosim.types.config import BioConfig, SimConfig


class TestBioConfig:
    def test_defaults(self, bio_config: BioConfig) -> None:
        assert bio_config.competition is True
        assert bio_config.cell_cycle is False
        assert bio_config.apoptosis is False
        assert bio_config.mapk is False
        assert bio_config.fba is False
        assert bio_config.replicator is False

    def test_override(self) -> None:
        cfg = BioConfig(competition=False, cell_cycle=True)
        assert cfg.competition is False
        assert cfg.cell_cycle is True

    def test_is_pydantic_model(self, bio_config: BioConfig) -> None:
        d = bio_config.model_dump()
        assert isinstance(d, dict)
        assert "competition" in d


class TestSimConfig:
    def test_defaults(self, sim_config: SimConfig) -> None:
        assert sim_config.max_companies == 50
        assert sim_config.tick_interval_ms == 1000
        assert sim_config.max_speed == 20
        assert sim_config.insolvent_ticks_to_death == 3

    def test_override(self) -> None:
        cfg = SimConfig(max_companies=100, tick_interval_ms=500)
        assert cfg.max_companies == 100
        assert cfg.tick_interval_ms == 500
