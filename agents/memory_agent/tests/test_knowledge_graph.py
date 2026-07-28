import tempfile
from pathlib import Path

import pytest

from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
from agents.memory_agent.schemas.models import EntityType, RelationType


@pytest.fixture
def kg(tmp_path):
    path = str(tmp_path / "test_kg.json")
    return KnowledgeGraph(persistence_path=path, max_nodes=100)


class TestEntityOperations:
    def test_add_entity(self, kg):
        node = kg.add_entity("COVID-19", EntityType.CONCEPT)
        assert node.entity_id
        assert node.name == "COVID-19"
        assert node.canonical_name == "covid-19"
        assert node.entity_type == EntityType.CONCEPT

    def test_add_entity_returns_existing(self, kg):
        n1 = kg.add_entity("COVID-19", EntityType.CONCEPT)
        n2 = kg.add_entity("COVID-19", EntityType.CONCEPT)
        assert n1.entity_id == n2.entity_id
        assert n2.access_count == 1

    def test_add_entity_with_properties(self, kg):
        node = kg.add_entity(
            "Pfizer", EntityType.ORGANIZATION, properties={"domain": "healthcare"}
        )
        assert node.properties["domain"] == "healthcare"

    def test_find_entity_by_name(self, kg):
        kg.add_entity("COVID-19", EntityType.CONCEPT)
        results = kg.find_entity_by_name("covid-19")
        assert len(results) == 1

    def test_find_entity_by_name_and_type(self, kg):
        kg.add_entity("COVID-19", EntityType.CONCEPT)
        kg.add_entity("COVID-19", EntityType.CLAIM)
        results = kg.find_entity_by_name("covid-19", EntityType.CONCEPT)
        assert len(results) == 1

    def test_remove_entity(self, kg):
        node = kg.add_entity("Test", EntityType.CONCEPT)
        assert kg.remove_entity(node.entity_id) is True
        assert kg.get_entity(node.entity_id) is None

    def test_remove_nonexistent_entity(self, kg):
        assert kg.remove_entity("nonexistent") is False

    def test_get_entity(self, kg):
        node = kg.add_entity("Test", EntityType.CONCEPT)
        retrieved = kg.get_entity(node.entity_id)
        assert retrieved is not None
        assert retrieved.name == "Test"


class TestEdgeOperations:
    def test_add_edge(self, kg):
        n1 = kg.add_entity("A", EntityType.CONCEPT)
        n2 = kg.add_entity("B", EntityType.CONCEPT)
        edge = kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        assert edge is not None
        assert edge.weight == 1.0

    def test_add_edge_with_nonexistent_node(self, kg):
        edge = kg.add_edge("a", "b", RelationType.SUPPORTS)
        assert edge is None

    def test_add_duplicate_edge_increments(self, kg):
        n1 = kg.add_entity("A", EntityType.CONCEPT)
        n2 = kg.add_entity("B", EntityType.CONCEPT)
        e1 = kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS, weight=0.5)
        e2 = kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        assert e1 is e2
        assert e2.evidence_count == 2
        assert e2.weight == 0.6

    def test_get_neighbors(self, kg):
        n1 = kg.add_entity("A", EntityType.CONCEPT)
        n2 = kg.add_entity("B", EntityType.CONCEPT)
        n3 = kg.add_entity("C", EntityType.CONCEPT)
        kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        kg.add_edge(n1.entity_id, n3.entity_id, RelationType.CONTRADICTS)

        neighbors = kg.get_neighbors(n1.entity_id, direction="out")
        assert len(neighbors) == 2

    def test_get_neighbors_filtered(self, kg):
        n1 = kg.add_entity("A", EntityType.CONCEPT)
        n2 = kg.add_entity("B", EntityType.CONCEPT)
        n3 = kg.add_entity("C", EntityType.CONCEPT)
        kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        kg.add_edge(n1.entity_id, n3.entity_id, RelationType.CONTRADICTS)

        neighbors = kg.get_neighbors(n1.entity_id, relation=RelationType.SUPPORTS)
        assert len(neighbors) == 1


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "test_kg.json")
        kg1 = KnowledgeGraph(persistence_path=path)
        n1 = kg1.add_entity("A", EntityType.CONCEPT)
        n2 = kg1.add_entity("B", EntityType.CONCEPT)
        kg1.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        kg1.save()

        kg2 = KnowledgeGraph(persistence_path=path)
        assert kg2.get_entity(n1.entity_id) is not None
        assert kg2.get_entity(n2.entity_id) is not None


class TestStats:
    def test_empty_graph_stats(self, kg):
        stats = kg.get_stats()
        assert stats.total_nodes == 0
        assert stats.total_edges == 0

    def test_populated_graph_stats(self, kg):
        kg.add_entity("A", EntityType.CONCEPT, properties={"domain": "test"})
        kg.add_entity("B", EntityType.SOURCE, properties={"domain": "test"})
        stats = kg.get_stats()
        assert stats.total_nodes == 2
        assert "test" in stats.domains_covered

    def test_apply_decay(self, kg):
        n1 = kg.add_entity("A", EntityType.CONCEPT)
        n2 = kg.add_entity("B", EntityType.CONCEPT)
        kg.add_edge(n1.entity_id, n2.entity_id, RelationType.SUPPORTS)
        decayed = kg.apply_decay()
        assert decayed == 1
