# LLM-name-tagging
LLM의 캐글데이터를 이용하여 해당하는 LLM의 이름을 맞추는 머신러닝 모델입니다

## B: EDA 및 특징 추출

- [작업 보고서와 팀원 사용 안내](docs/B-EDA-and-features.md)
- 공통 문체 특징 32개, 문자 TF-IDF, 결합 특징: `features/`
- 학습 데이터 분석 결과와 그래프: `results/b_eda/`

```bash
python -m pip install -r requirements-b.txt
python -m features.eda
python -m features.build --kind style
```
