import math

import numpy as np
import pytest

from faceapp.matching import (age_score, attach_face_probs, cosine_similarity, directional_score, hobby_score,
                              is_eligible, match_score, preference_from_reference_probs, preference_vector,
                              recommend_top_n)

W = {"weights": {"face": 0.6, "age": 0.25, "hobby": 0.15}, "age_tolerance": 3, "bidirectional": True, "top_n": 5}


def person(pid, gender="F", age=22, probs=(0.25, 0.25, 0.25, 0.25), hobbies=("영화",), pref=None):
    return {"id": pid, "gender": gender, "age": age, "face_probs": list(probs), "hobbies": list(hobbies),
            "preference": pref or {}}


# ---------------------------------------------------------------- components
def test_cosine_basic():
    assert cosine_similarity([1, 0, 0, 0], [1, 0, 0, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine_similarity([0, 0, 0, 0], [1, 0, 0, 0]) == 0.0  # 영벡터 안전 처리


def test_cosine_scale_invariant():
    a = np.array([0.7, 0.2, 0.05, 0.05])
    assert cosine_similarity(a, [0.7, 0.3, 0, 0]) == pytest.approx(cosine_similarity(a * 10, [7, 3, 0, 0]))


def test_preference_vector_dict_and_list():
    v = preference_vector({"dog": 3, "Cat": 1})
    assert v.tolist() == pytest.approx([0.75, 0.25, 0, 0])
    assert preference_vector([0, 0, 2, 2]).tolist() == pytest.approx([0, 0, 0.5, 0.5])
    assert preference_vector(None) is None
    assert preference_vector({}) is None


@pytest.mark.parametrize("bad", [{"hamster": 1}, [1, 2, 3], {"dog": -1}])
def test_preference_vector_invalid(bad):
    with pytest.raises(ValueError):
        preference_vector(bad)


def test_preference_from_reference_probs():
    v = preference_from_reference_probs([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert v.tolist() == pytest.approx([0.5, 0.5, 0, 0])


@pytest.mark.parametrize("age,expected", [(22, 1.0), (21, 1.0), (25, 1.0), (26, 2 / 3), (28, 0.0), (19, 1 / 3), (40, 0.0)])
def test_age_score_range_and_tolerance(age, expected):
    assert age_score(age, 21, 25, tolerance=3) == pytest.approx(expected)


def test_age_score_open_and_missing():
    assert age_score(30, age_min=21) == 1.0
    assert age_score(22, None, None) is None
    assert age_score(None, 20, 25) is None
    assert age_score(26, 21, 25, tolerance=0) == 0.0


def test_hobby_jaccard():
    assert hobby_score(["영화", "러닝"], ["러닝", "독서"]) == pytest.approx(1 / 3)
    assert hobby_score(["Movie "], ["movie"]) == 1.0  # 대소문자/공백 정규화
    assert hobby_score([], ["영화"]) is None


# ------------------------------------------------------------------ scoring
def test_directional_score_weighted_sum():
    viewer = person("a", pref={"face_type": {"dog": 1}, "age_min": 21, "age_max": 25}, hobbies=["영화", "러닝"])
    cand = person("b", age=23, probs=(1, 0, 0, 0), hobbies=["러닝"])
    score, parts = directional_score(viewer, cand, W)
    assert parts == {"face": pytest.approx(1.0), "age": 1.0, "hobby": pytest.approx(0.5)}
    assert score == pytest.approx((0.6 * 1 + 0.25 * 1 + 0.15 * 0.5) / 1.0)


def test_missing_components_are_renormalized():
    viewer = person("a", pref={"face_type": {"cat": 1}}, hobbies=[])  # 나이 선호·취미 없음 → 얼굴만
    cand = person("b", probs=(0, 1, 0, 0))
    score, parts = directional_score(viewer, cand, W)
    assert parts["age"] is None and parts["hobby"] is None
    assert score == pytest.approx(1.0)


def test_face_preference_ranks_matching_type_higher():
    viewer = person("a", pref={"face_type": {"fox": 1}})
    fox = person("fox", probs=(0.05, 0.05, 0.1, 0.8))
    dog = person("dog", probs=(0.8, 0.1, 0.05, 0.05))
    assert directional_score(viewer, fox, W)[0] > directional_score(viewer, dog, W)[0]


def test_bidirectional_is_symmetric_average():
    a = person("a", pref={"face_type": {"dog": 1}}, probs=(0, 1, 0, 0))
    b = person("b", pref={"face_type": {"dog": 1}}, probs=(1, 0, 0, 0))
    s_ab, detail = match_score(a, b, W)
    s_ba, _ = match_score(b, a, W)
    assert s_ab == pytest.approx(s_ba)
    assert s_ab == pytest.approx((detail["a_to_b"] + detail["b_to_a"]) / 2)
    one_way, _ = match_score(a, b, {**W, "bidirectional": False})
    assert one_way == pytest.approx(detail["a_to_b"])


def test_eligibility_excludes_self_and_gender_filter():
    a = person("a", gender="F", pref={"gender": "M"})
    assert not is_eligible(a, a)
    assert not is_eligible(a, person("b", gender="F"))
    assert is_eligible(a, person("c", gender="M"))


# ---------------------------------------------------------------- top-n
def test_recommend_top_n_order_and_filters():
    viewer = person("v", gender="F", pref={"face_type": {"rabbit": 1}, "gender": "M"})
    cands = [
        person("m1", gender="M", probs=(0.1, 0.1, 0.7, 0.1)),
        person("m2", gender="M", probs=(0.0, 0.0, 1.0, 0.0)),
        person("m3", gender="M", probs=(0.7, 0.1, 0.1, 0.1)),
        person("f1", gender="F", probs=(0.0, 0.0, 1.0, 0.0)),  # 성별 필터로 제외
    ]
    recs = recommend_top_n([viewer, *cands], {**W, "bidirectional": False}, top_n=2)
    ids = [c["id"] for c in recs["v"]]
    assert ids == ["m2", "m1"]
    assert len(recs["v"]) == 2


def test_recommend_top_n_bidirectional_requires_mutual_eligibility():
    a = person("a", gender="F", pref={"gender": "M"})
    b = person("b", gender="M", pref={"gender": "M"})  # b는 남성을 원함 → a는 b의 대상이 아님
    recs = recommend_top_n([a, b], W)
    assert recs["a"] == [] and recs["b"] == []


def test_recommend_ties_broken_by_id():
    v = person("v", pref={"face_type": {"dog": 1}})
    recs = recommend_top_n([v, person("z"), person("b")], {**W, "bidirectional": False})
    assert [c["id"] for c in recs["v"]] == ["b", "z"]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        recommend_top_n([person("a"), person("a")], W)


def test_attach_face_probs():
    people = [{"id": "a", "photo": "photos/a.jpg"}, {"id": "b", "photo": "b.jpg"}]
    out = attach_face_probs(people, {"a.jpg": [1, 0, 0, 0], "b.jpg": [0, 1, 0, 0]})
    assert out[0]["face_probs"] == [1, 0, 0, 0] and "face_probs" not in people[0]
    with pytest.raises(KeyError):
        attach_face_probs([{"id": "c", "photo": "c.jpg"}], {})
