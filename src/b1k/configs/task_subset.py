"""12개 태스크 subset 정의 및 전역/로컬 id 변환 유틸리티 (task_subset.py)

이 파일의 역할:
    원래 BEHAVIOR-1K 데이터셋에는 수십 개의 태스크가 있지만,
    이번 실험에서는 그 중 12개만 선택해서 학습/평가한다.

    문제: 데이터셋과 환경(OmniGibson)은 여전히 원래 전역 task id(0~N)를 사용한다.
    모델 내부: task_embeddings 테이블이 12개 행만 가지므로 0~11 범위의 로컬 id를 써야 한다.

    따라서 데이터 파이프라인에서 전역 id → 로컬 id 변환이 반드시 필요하다.
    이 파일이 그 변환 테이블과 변환 함수를 제공한다.

사용 흐름:
    데이터셋(전역 task_id=5)
        ↓ map_global_to_local(5) → 로컬 id = 3
    모델(task_embeddings[3])
        ↓ 추론 완료 후 로깅/평가
    평가 환경(전역 task_id = LOCAL_TO_GLOBAL[3] = 5)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 선택한 태스크 목록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 실제로 학습/평가에 사용할 12개 태스크의 원래 전역 task id 목록.
# 이 순서가 로컬 id(0~11)로 매핑되는 순서와 동일하다.
#
# 예: SELECTED_TASKS[0] = 0   -> 전역 id 0번 = 로컬 id 0번
#     SELECTED_TASKS[2] = 5   -> 전역 id 5번 = 로컬 id 2번
#
# 선택 기준: 데이터 다양성, episode 수, 태스크 복잡도 등을 고려해 선별한 12개.
SELECTED_TASKS: list[int] = [0, 1, 5, 11, 12, 19, 22, 31, 39, 40, 45, 46]

# 빠른 멤버십 확인을 위한 set 버전.
# "if task_id in SELECTED_TASKS_SET" 형태로 O(1) 조회에 사용한다.
SELECTED_TASKS_SET: set[int] = set(SELECTED_TASKS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# id 변환 테이블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 전역 task id → 모델 내부 로컬 id (0~11) 변환 딕셔너리.
# enumerate(SELECTED_TASKS)를 역으로 만들어서 전역 → 로컬 방향으로 조회한다.
#
# 예: GLOBAL_TO_LOCAL = {
#     0: 0, 1: 1, 5: 2, 11: 3, 12: 4, 19: 5,
#     22: 6, 31: 7, 39: 8, 40: 9, 45: 10, 46: 11
# }
GLOBAL_TO_LOCAL: dict[int, int] = {
    global_id: local_id
    for local_id, global_id in enumerate(SELECTED_TASKS)
}

# 모델 내부 로컬 id (0~11) → 전역 task id 변환 딕셔너리.
# GLOBAL_TO_LOCAL의 역방향 매핑이다.
# 추론 결과를 평가 환경에 전달하거나 로그를 남길 때 사용한다.
#
# 예: LOCAL_TO_GLOBAL = {
#     0: 0, 1: 1, 2: 5, 3: 11, 4: 12, 5: 19,
#     6: 22, 7: 31, 8: 39, 9: 40, 10: 45, 11: 46
# }
LOCAL_TO_GLOBAL: dict[int, int] = {
    local_id: global_id
    for global_id, local_id in GLOBAL_TO_LOCAL.items()
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 변환 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def map_global_to_local(global_task_id: int) -> int:
    """전역 task id를 모델 내부 로컬 task id(0~11)로 변환한다.

    데이터 로더나 환경 wrapper에서 데이터셋의 task_index 필드를
    모델에 넘기기 전에 호출해야 한다.

    Args:
        global_task_id: 원본 BEHAVIOR-1K 데이터셋의 task index (0~49 범위).

    Returns:
        모델 내부에서 사용하는 로컬 task index (0~11 범위).

    Raises:
        KeyError: global_task_id가 선택한 12개 태스크 중 하나가 아닐 때.
            → 데이터 필터링이 제대로 적용되었는지 확인해야 한다.

    예:
        map_global_to_local(5) → 2  (전역 5번 → 로컬 2번)
        map_global_to_local(99) → KeyError 발생
    """
    if global_task_id not in GLOBAL_TO_LOCAL:
        raise KeyError(
            f"전역 task id {global_task_id}는 선택한 12개 태스크 subset에 없습니다. "
            f"허용된 전역 id 목록: {SELECTED_TASKS}"
        )
    return GLOBAL_TO_LOCAL[global_task_id]


def map_local_to_global(local_task_id: int) -> int:
    """모델 내부 로컬 task id(0~11)를 원래 전역 task id로 변환한다.

    주로 추론 결과를 로깅하거나 평가 환경에 task 정보를 전달할 때 사용한다.

    Args:
        local_task_id: 모델 내부 로컬 task index (0~11 범위).

    Returns:
        원본 BEHAVIOR-1K 데이터셋의 전역 task index.

    Raises:
        KeyError: local_task_id가 0~11 범위를 벗어날 때.

    예:
        map_local_to_global(3) → 11  (로컬 3번 → 전역 11번)
        map_local_to_global(12) → KeyError 발생
    """
    if local_task_id not in LOCAL_TO_GLOBAL:
        raise KeyError(
            f"로컬 task id {local_task_id}는 유효하지 않습니다. "
            f"허용 범위: 0~{len(SELECTED_TASKS) - 1}"
        )
    return LOCAL_TO_GLOBAL[local_task_id]
