"""딥러닝 얼굴 유형 분류 기반 소개팅 매칭 프로토타입."""

# 클래스 순서의 유일한 기준. 모델 출력 index, 확률 벡터, 선호 벡터 모두 이 순서를 따른다.
CLASSES = ["dog", "cat", "rabbit", "fox"]
CLASS_KO = {"dog": "강아지상", "cat": "고양이상", "rabbit": "토끼상", "fox": "여우상"}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
