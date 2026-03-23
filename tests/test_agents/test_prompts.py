from __future__ import annotations

from biosim.agents.prompts import DEPARTMENT_DOMAINS, build_system_prompt


class TestDepartmentDomains:
    def test_all_12_departments_have_domains(self):
        assert len(DEPARTMENT_DOMAINS) == 12

    def test_all_indices_present(self):
        for i in range(12):
            assert i in DEPARTMENT_DOMAINS

    def test_each_domain_has_required_keys(self):
        for idx, domain in DEPARTMENT_DOMAINS.items():
            assert "name" in domain, f"Missing name for dept {idx}"
            assert "domain_description" in domain, f"Missing domain_description for dept {idx}"
            assert "authority_boundaries" in domain, f"Missing authority_boundaries for dept {idx}"

    def test_department_names_match_expected(self):
        expected = [
            "Finance", "R&D", "Distribution", "Production", "Sales", "Marketing",
            "HR", "Executive", "Customer Service", "Legal", "IT", "Procurement",
        ]
        for idx, name in enumerate(expected):
            assert DEPARTMENT_DOMAINS[idx]["name"] == name


class TestBuildSystemPrompt:
    def test_contains_company_name(self):
        prompt = build_system_prompt("TestCorp", "Finance", 0)
        assert "TestCorp" in prompt

    def test_contains_dept_name(self):
        prompt = build_system_prompt("TestCorp", "Finance", 0)
        assert "Finance" in prompt

    def test_contains_domain_description(self):
        prompt = build_system_prompt("TestCorp", "Finance", 0)
        assert "Cash flow management" in prompt

    def test_contains_authority_boundaries(self):
        prompt = build_system_prompt("TestCorp", "Finance", 0)
        assert "Cannot hire/fire" in prompt

    def test_contains_json_format_instructions(self):
        prompt = build_system_prompt("TestCorp", "R&D", 1)
        assert '"action"' in prompt
        assert '"confidence"' in prompt

    def test_all_departments_build_successfully(self):
        for idx, domain in DEPARTMENT_DOMAINS.items():
            prompt = build_system_prompt("Corp", domain["name"], idx)
            assert domain["name"] in prompt
            assert "Corp" in prompt
