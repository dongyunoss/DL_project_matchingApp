from scripts.sample_celeba_candidates import sample_one_per_identity


def test_one_image_per_identity_and_n():
    id_map = {f"{i:06d}.jpg": str(i % 7) for i in range(50)}
    rows = sample_one_per_identity(id_map, n=5, seed=0)
    assert len(rows) == 5
    assert len({ident for _, ident in rows}) == 5
    assert all(id_map[f] == ident for f, ident in rows)
    assert rows == sample_one_per_identity(id_map, n=5, seed=0)  # 재현성


def test_filter_applied_before_sampling():
    id_map = {"a.jpg": "1", "b.jpg": "1", "c.jpg": "2"}
    rows = sample_one_per_identity(id_map, n=None, keep=lambda f: f != "a.jpg")
    assert sorted(rows) == [("b.jpg", "1"), ("c.jpg", "2")]
