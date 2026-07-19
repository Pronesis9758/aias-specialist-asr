# Data governance and human review

## 허용 데이터

- 직접 작성한 제조 시나리오 문장과 합성 음성
- 명시적으로 공개된 음성 데이터
- 회사 승인 절차를 거친 비식별 데이터

현재 공개 데모는 `kresnik/zeroth_korean`의 CC BY 4.0 고정 리비전을 사용한다. 생성된
manifest와 `dataset_provenance.json`에 저장소, commit SHA, 라이선스, 원본 ID, split을
기록하고 `DATASET_NOTICE.md`의 저작자 표시를 유지한다.

## 금지 또는 별도 승인 데이터

- 작업자/고객의 개인정보가 포함된 음성
- 설비 식별자, 생산량, 불량 원인 등 영업비밀이 포함된 원본
- 외부 반출이 금지된 제조 문서와 로그

## 필수 manifest 필드

각 샘플은 출처와 승인 상태를 기록해야 한다. `consent_status`가 `approved`, `synthetic`,
`public`이 아니면 파이프라인이 실행을 중단한다.

## 보고서 표현

AI 도구는 반복 실험과 코드·리포트 생성을 자동화한 개발 도구로 기록한다. 문제 정의, 데이터 구성,
평가 지표, 오류 분석과 최종 개선 방향은 과제 수행자가 정의하고 검증했다는 역할 구분을 유지한다.
