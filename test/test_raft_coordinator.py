"""Tests for Raft consensus engine and cluster registry (Phase 2)."""

import json
import os
import threading
import time
from unittest.mock import patch

import pytest

from services.raft_coordinator import (
    RaftState,
    RaftNode,
    ClusterRegistry,
    RaftCoordinator,
)


# ---- RaftState tests -------------------------------------------------------

def test_raft_state_initial_follower():
    state = RaftState(node_id="n1", peers=["n2", "n3"])
    assert state.role == "follower"
    assert state.current_term == 0
    assert state.voted_for is None
    assert state.commit_index == -1
    assert state.last_applied == -1


def test_become_follower():
    state = RaftState("n1", ["n2"])
    state.become_follower(term=3, leader_id="n2")
    assert state.role == "follower"
    assert state.current_term == 3
    assert state.leader_id == "n2"
    assert state.voted_for is None


def test_become_candidate():
    state = RaftState("n1", ["n2", "n3"])
    state.become_candidate()
    assert state.role == "candidate"
    assert state.current_term == 1
    assert state.voted_for == "n1"
    assert state.leader_id is None


def test_become_leader():
    state = RaftState("n1", ["n2", "n3"])
    state.become_candidate()
    state.become_leader()
    assert state.role == "leader"
    assert state.leader_id == "n1"
    assert state.next_index == {"n2": 0, "n3": 0}
    assert state.match_index == {"n2": -1, "n3": -1}


def test_append_entry():
    state = RaftState("n1", ["n2"])
    state.current_term = 2
    idx = state.append_entry(2, {"op": "INSERT", "vector_id": "v1"})
    assert idx == 0
    assert len(state.log) == 1
    assert state.log[0]["term"] == 2
    assert state.log[0]["command"]["op"] == "INSERT"

    idx2 = state.append_entry(2, {"op": "DELETE", "vector_id": "v2"})
    assert idx2 == 1
    assert len(state.log) == 2


def test_append_multiple_entries():
    state = RaftState("n1", ["n2"])
    for i in range(10):
        state.append_entry(1, {"op": "INSERT", "i": i})
    assert len(state.log) == 10


# ---- ClusterRegistry tests -------------------------------------------------

def test_cluster_register_node(tmp_path):
    path = str(tmp_path / "cluster.json")
    reg = ClusterRegistry(local_node_id="n1", config_path=path)
    assert reg.register_node("n2", "localhost:8002") is True
    assert reg.register_node("n2", "localhost:8002") is False  # duplicate
    assert len(reg.get_active_nodes()) == 1


def test_cluster_remove_node(tmp_path):
    path = str(tmp_path / "cluster.json")
    reg = ClusterRegistry(local_node_id="n1", config_path=path)
    reg.register_node("n2", "localhost:8002")
    assert reg.remove_node("n2") is True
    assert len(reg.get_active_nodes()) == 0
    assert reg.remove_node("n2") is False


def test_cluster_get_peer_addresses_excludes_self(tmp_path):
    path = str(tmp_path / "cluster.json")
    reg = ClusterRegistry(local_node_id="n1", config_path=path)
    reg.register_node("n1", "localhost:8001")
    reg.register_node("n2", "localhost:8002")
    reg.register_node("n3", "localhost:8003")
    peers = reg.get_peer_addresses()
    assert "localhost:8001" not in peers
    assert "localhost:8002" in peers
    assert "localhost:8003" in peers


def test_cluster_mark_health(tmp_path):
    path = str(tmp_path / "cluster.json")
    reg = ClusterRegistry(local_node_id="n1", config_path=path)
    reg.register_node("n2", "localhost:8002")
    reg.mark_health("n2", False)
    assert len(reg.get_active_nodes()) == 0
    reg.mark_health("n2", True)
    assert len(reg.get_active_nodes()) == 1


def test_cluster_persistence(tmp_path):
    path = str(tmp_path / "cluster.json")
    reg1 = ClusterRegistry(local_node_id="n1", config_path=path)
    reg1.register_node("n2", "localhost:8002")
    reg1.register_node("n3", "localhost:8003")

    reg2 = ClusterRegistry(local_node_id="n1", config_path=path)
    assert len(reg2.get_active_nodes()) == 2


# ---- RaftCoordinator tests -------------------------------------------------

def test_coordinator_init(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2", "n3"], data_dir=d)
    assert coord.node_id == "n1"
    assert coord.state.role == "follower"
    assert coord.state.peers == ["n2", "n3"]


def test_coordinator_propose_as_leader(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    # Manually become leader for testing
    coord.state.become_leader()
    result = coord.propose({"op": "INSERT", "vector_id": "v1"})
    assert result["success"] is True
    assert result["index"] == 0
    assert result["leader_id"] == "n1"
    assert len(coord.state.log) == 1


def test_coordinator_propose_as_follower(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    result = coord.propose({"op": "INSERT"})
    assert result["success"] is False
    assert "Not the leader" in result["message"]


def test_coordinator_multiple_proposals(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    coord.state.become_leader()
    for i in range(5):
        coord.propose({"op": "INSERT", "i": i})
    assert len(coord.state.log) == 5
    assert coord.state.commit_index >= 0


def test_coordinator_apply_fn_called(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    applied = []

    def apply_fn(cmd):
        applied.append(cmd)

    coord.set_apply_fn(apply_fn)
    coord.state.become_leader()
    coord.propose({"op": "INSERT", "id": "v1"})
    # Give time for async apply
    time.sleep(0.2)
    assert len(applied) >= 1
    assert applied[0]["op"] == "INSERT"


def test_coordinator_read_state(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    result = coord.read()
    assert result["node_id"] == "n1"
    assert "role" in result
    assert "current_term" in result
    assert "peers" in result


def test_coordinator_get_status(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2", "n3"], data_dir=d)
    status = coord.get_status()
    assert status["node_id"] == "n1"
    assert "role" in status
    assert "active_nodes" in status
    assert "log_size" in status


def test_coordinator_leader_election(tmp_path):
    """Simplified test that the election loop can transition to leader."""
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    coord.start()
    time.sleep(2.5)  # Wait for election timeout (1.5s default)
    # The simplified implementation will become leader
    # since _request_vote always returns True
    assert coord.state.role == "leader"
    coord.stop()


def test_coordinator_advance_commit_index(tmp_path):
    d = str(tmp_path / "raft")
    coord = RaftCoordinator(node_id="n1", address="localhost:8001",
                            peers=["n2"], data_dir=d)
    coord.state.become_leader()
    coord.state.log = [{"term": 1, "command": {"op": "INSERT"}} for _ in range(5)]
    coord.state.match_index = {"n2": 4}
    orig_commit = coord.state.commit_index
    coord._advance_commit_index()
    assert coord.state.commit_index > orig_commit or coord.state.commit_index >= 0


def test_raft_node_dataclass():
    node = RaftNode(node_id="n1", address="localhost:8001")
    assert node.node_id == "n1"
    assert node.is_active is True

    node_inactive = RaftNode(node_id="n2", address="localhost:8002", is_active=False)
    assert node_inactive.is_active is False
