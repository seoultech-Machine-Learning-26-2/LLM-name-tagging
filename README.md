# LLM-name-tagging
LLM의 캐글데이터를 이용하여 해당하는 LLM의 이름을 맞추는 머신러닝 모델입니다

## 팀 작업 문서

- [전체 문서와 A·B·C·D 진행 기록](docs/README.md)
- [A: 데이터 분할 및 관리](docs/A-data-split.md)
- [C: 평가 코드와 베이스라인](docs/C-evaluation-and-baseline.md)
- [D: 모델 조사 및 실험 설계](docs/D-model-research-and-design.md)
- [실험 결과 템플릿](docs/experiment-result-template.md)
- [다음 단계 작업 계획](docs/next-tasks-A-B-C-D.md)

## B: EDA 및 특징 추출

- [작업 보고서와 팀원 사용 안내](docs/B-EDA-and-features.md)
- 공통 문체 특징 32개, 문자 TF-IDF, 결합 특징: `features/`
- 학습 데이터 분석 결과와 그래프: `results/b_eda/`

```bash
python -m pip install -r requirements-b.txt
python -m features.eda
python -m features.build --kind style
```
