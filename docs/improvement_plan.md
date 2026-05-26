# Future Work

이 문서는 중간발표 PPT의 `향후 계획` 내용을 repo용으로 정리한 것입니다.

## 1. 학습 관련

- 학습 step 증가를 통한 장기 수렴 성능 분석
- 다양한 seed 기반 반복 실험을 통한 모델 재현성 검증
- Task 수 확장을 통한 multi-task generalization 성능 분석

## 2. 추가 VLA 기술

- **MeM (Multi-Scale Embodied Memory)**: short-term / long-term memory 기반 long-horizon task 안정성 향상
- **RD-VLA (Recurrent-Depth VLA)**: latent action plan을 반복적으로 refinement하여 long-horizon task에서 더 정밀한 action trajectory 생성
- **π*0.6-lite Recovery Learning**: 실패 rollout과 stage/value 정보를 활용하여 OOD 상태에서 recovery behavior 및 rollout robustness 향상

## Notes

현재 repo의 lightweight baseline은 1등팀 대비 `step 200,000 -> 70,000`, `batch size 2048 -> 16`, `flow sample 15 -> 1`로 줄인 실험입니다. 후속 작업은 먼저 학습 규모와 재현성을 확인하고, 이후 long-horizon 안정성 및 OOD recovery를 보강하는 방향으로 진행합니다.
