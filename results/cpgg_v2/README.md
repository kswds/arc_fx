# CPGG-v2 실행 방법

모든 명령은 `src/`에서 실행한다. 한 번에 프로세스 1개, CPU 스레드 2개.

| 단계 | 명령 | 시간 / 메모리 |
|---|---|---|
| 단위 테스트 | `PYTHONPATH=. ../.venv/bin/python ../tests/test_cpgg_v2.py` | 수 초 |
| 합성 점검 | `PYTHONPATH=. OMP_NUM_THREADS=2 ../.venv/bin/python -W ignore -m cpgg_v2.synthetic_checks --only D,A,B,C --tag attempt0` | 약 2분 |
| 실제 데이터 | `PYTHONPATH=. OMP_NUM_THREADS=2 ../.venv/bin/python -W ignore -m cpgg_v2.experiment --seed 0` | 약 85초, 1.9 GB |
| 체크포인트 표 | `PYTHONPATH=. ../.venv/bin/python -m cpgg_v2.report --seed 0 --synthetic attempt0` | 수 초 |

test 평가 명령은 최종 설정 확정 후 추가한다(현재 미제공, test 미사용).
