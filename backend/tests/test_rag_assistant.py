from app.rag.store import PolicyStore


def test_policy_retrieval_grounded():
    store = PolicyStore()
    hits = store.search("What is the procedure for a vehicle entering Restricted Zone A?")
    assert hits and hits[0]["source"] == "vehicle_intrusion_sop.md"


def test_policy_no_match_returns_empty():
    assert PolicyStore().search("quantum banana helicopter") == []


def test_answer_chunk_contains_actual_procedure_not_just_a_title():
    top = PolicyStore().search("What is the procedure for a vehicle entering Restricted Zone A?")[0]
    assert "operator acknowledges the alert" in top["text"]


def test_light_stemming_matches_word_variants():
    top = PolicyStore().search("what happens when a dismissal is made")[0]
    assert top["source"] == "inspection_and_reporting.md" and top["heading"] == "Incident review"
