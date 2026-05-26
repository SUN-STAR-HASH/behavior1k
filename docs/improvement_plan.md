# Improvement Plan

이 문서는 70k lightweight baseline 평가 이후의 개선 방향을 정리한 것입니다.

## 핵심 결론

현재 낮은 점수는 단순히 모델이 관측을 전혀 이해하지 못해서라기보다, 1등팀 구조를 유지하더라도 학습 규모를 `200k step / batch 2048 / multi-flow sample 15`에서 `70k step / batch 16 / multi-flow sample 1`로 줄이면 BEHAVIOR-1K의 긴 조작 task를 충분히 학습하기 어렵다는 신호로 봅니다.

- **학습량 부족**: 70k step은 빠른 baseline으로 의미가 있지만, 200k reference보다 수렴 시간이 짧습니다.
- **작은 batch의 분산**: batch 16은 batch 2048 대비 gradient variance가 커질 수 있습니다.
- **single flow sample의 분산**: multi-flow sample 1은 15보다 flow matching 학습 신호가 불안정할 수 있습니다.
- **실패 복구 부족**: 버튼 누르기, grasp, object alignment가 한 번 어긋나면 다시 회복하지 못합니다.
- **instance 다양성 취약**: 한 task 안에서도 instance 배치가 달라지면 Q-score가 쉽게 0으로 떨어질 수 있습니다.

## 1. 학습량 확장

다음 실험은 3주 기준으로 180k~210k step까지 확장해 1등팀 reference의 200k step에 가까운 수렴을 확인합니다.

| 단계 | Step 범위 | 목표 |
| --- | --- | --- |
| 1주차 | 0~70k | 현재 baseline 재현, stage tracking 안정화 |
| 2주차 | 70k~140k | 실패가 많은 task/stage oversampling |
| 3주차 | 140k~200k/210k | hard instance 집중 학습, best checkpoint 선택 |

권장 운영:

- checkpoint는 10k 또는 20k마다 저장
- evaluation은 20k 또는 30k마다 부분 평가
- 최종 선택 기준은 loss가 아니라 실제 Q-score / stage completion

## 2. Batch / Flow Sample Ablation

학습 규모를 한 번에 1등팀 수준으로 되돌리기 어렵다면, 비용이 낮은 축부터 순차적으로 비교합니다.

| 실험 | 목표 |
| --- | --- |
| batch 16 유지, step 200k | 학습량 부족 영향 확인 |
| gradient accumulation | effective batch 확대 효과 확인 |
| `num_flow_samples=3` | single-flow 대비 안정성 확인 |
| `num_flow_samples=5` | 비용과 점수 사이의 중간 지점 확인 |
| `num_flow_samples=15` | 1등팀 reference에 가까운 flow sampling 확인 |

## 3. MeM-lite

큰 memory architecture를 바로 붙이기보다, baseline에 맞는 작은 memory signal부터 추가합니다.

| 추가 입력 | 의미 |
| --- | --- |
| stage history embedding | 지금까지 어떤 stage를 지나왔는지 |
| no-progress flag | 일정 시간 동안 stage가 변하지 않았는지 |
| last failure type embedding | 최근 실패가 grasp, contact, alignment 중 무엇인지 |
| current subgoal id/text | 지금 해결해야 할 작은 목표 |

목표는 대형 LLM planner를 매 inference마다 호출하지 않고도 long-horizon 진행 상태를 모델 입력에 제공하는 것입니다.

## 4. Recovery Learning

성공 demo만 모방하는 방식은 실패 이후 상태를 잘 다루지 못합니다. 다음 항목을 로그로 남기고 재학습에 활용합니다.

- gripper-object 거리
- gripper open/close 상태
- q-score 변화량
- stage 변화량
- no-progress 구간
- 실패 후 사람이 고치거나 policy가 회복한 action

후속 실험:

- recovery action oversampling
- Q-score / stage progress 기반 value head
- 실패로 이어진 action down-weighting

## 5. Instance Robustness

평가를 task 평균 하나로만 보지 않고 instance 단위로 관리합니다.

- task별 평균 Q-score와 instance별 Q-score를 함께 저장
- Q-score가 0인 instance를 hard instance로 표시
- hard instance를 다음 학습에서 더 자주 sampling
- 시작 pose, object 위치, camera noise, action noise를 조금씩 randomization
- validation instance는 training instance와 분리

## 6. Later Ablations

아래 기술은 baseline을 먼저 안정화한 뒤 하나씩 켜거나 끄면서 비용 대비 효과를 비교합니다.

- correlated noise
- FAST auxiliary loss
- KV transform
- knowledge insulation
- RD-VLA-lite recurrent action refinement

최종 목표는 1등팀 전체 모델을 그대로 복제하는 것이 아니라, 어떤 요소가 실제 성능 향상에 필요한지 단계적으로 확인하고 더 가벼운 BEHAVIOR-1K baseline을 만드는 것입니다.
