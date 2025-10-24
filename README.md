# Multimodal Emotion Classification (Speech · Text · Image)

> 음성·텍스트·이미지 기반 멀티모달 감정 인식 모델 개발 및 서비스 배포
> 기간: 2023.01 – 2023.06

---

## Summary

* 자사 감정 데이터 + AI Hub 음성 데이터 기반 **멀티모달 감정 모델 성능 개선**
* 감정 클래스 체계를 34→7종으로 재구성하여 **실사용 적합 데이터셋 구축**
* 모델/API 통합으로 **플랫폼 서비스에 실제 배포**

---

## Technical Highlights

* 감정 데이터 추가 수집/정제 및 **전처리 파이프라인 구축**
* Wav2Vec2 기반 **음성 감정 모델 학습**
* MFB(Multi-modal Factorized Bilinear Pooling) 기반 **Late Fusion 멀티모달 모델 구현**
* API 개발 및 서비스 플랫폼 배포로 **실사용 기능 개선**

---

## Visual Examples (to be inserted)

* 멀티모달 입력 구조 다이어그램 (`docs/figures/pipeline.png`)
* 감정 분포/리라벨링 예시 (`docs/figures/relabeling.png`)
* API 응답/사용 시나리오 캡처 (`docs/figures/api_demo.png`)

---

## Disclosure

* 감정 데이터 및 모델 가중치는 비공개
* 본 저장소는 구현 개요 및 구조 문서용으로 구성

---

## Contact

이현희 / AI Research Engineer
[fly4hyun@naver.com](mailto:fly4hyun@naver.com)
