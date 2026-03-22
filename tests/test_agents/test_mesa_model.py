from biosim.agents.mesa_model import BioSimModel
from biosim.types.config import BioConfig, SimConfig


class TestBioSimModelCreation:
    def test_default_creation(self) -> None:
        model = BioSimModel()
        assert model.state_manager.tick_count == 0
        assert len(model.agents_list) == 0

    def test_add_company_creates_agent(self) -> None:
        model = BioSimModel()
        agent = model.add_company("TestCo", "#FF0000", size="medium")

        assert agent.state_index == 0
        assert agent.name == "TestCo"
        assert agent.is_alive
        assert len(model.agents_list) == 1


class TestModelStep:
    def test_step_advances_tick_count(self) -> None:
        model = BioSimModel()
        model.add_company("A", "#F00")
        model.step()
        assert model.state_manager.tick_count == 1

    def test_step_stores_snapshot(self) -> None:
        model = BioSimModel()
        model.add_company("A", "#F00")
        model.step()
        assert "n_active" in model.last_snapshot
        assert model.last_snapshot["n_active"] == 1

    def test_step_records_history(self) -> None:
        model = BioSimModel()
        model.add_company("A", "#F00")
        model.step()
        assert len(model.state_manager.history) == 1


class TestDataCollector:
    def test_datacollector_records_metrics(self) -> None:
        model = BioSimModel()
        model.add_company("A", "#F00", size="medium")
        model.add_company("B", "#0F0", size="small")
        model.step()

        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 1
        assert model_data["n_active_companies"].iloc[0] == 2
        assert model_data["total_market_size"].iloc[0] > 0


class TestMultiCompanyInteraction:
    def test_ten_ticks_with_three_companies(self) -> None:
        config = SimConfig()
        bio = BioConfig(competition=True)
        model = BioSimModel(bio_config=bio, sim_config=config)

        model.add_company("Alpha", "#E74C3C", size="large")
        model.add_company("Beta", "#3498DB", size="medium")
        model.add_company("Gamma", "#2ECC71", size="small")

        for _ in range(10):
            model.step()

        assert model.state_manager.tick_count == 10
        assert len(model.state_manager.history) == 10
        assert model.last_snapshot["n_active"] >= 1

        model_data = model.datacollector.get_model_vars_dataframe()
        assert len(model_data) == 10

    def test_companies_have_different_trajectories(self) -> None:
        model = BioSimModel(bio_config=BioConfig(competition=True))
        model.add_company("Big", "#F00", size="large")
        model.add_company("Small", "#0F0", size="small")

        for _ in range(5):
            model.step()

        state = model.state_manager.state
        assert state.firm_size[0] != state.firm_size[1]
