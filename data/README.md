# Data contract

- `sample/`: 공개 가능한 합성 fixture. CI와 smoke test에서만 사용합니다.
- `domain_terms/`: 현업 검수를 거친 canonical 제조 용어와 ASR 오인식 alias입니다.
- `raw_audio/`: 공개 가능 음성만 임시로 둘 수 있으며 기본적으로 Git에서 제외됩니다.
- `private/`: 실제 음성/전사 manifest 위치. 전체 디렉터리가 Git에서 제외됩니다.

실제 데이터 manifest의 `consent_status`는 최소 `approved`, `synthetic`, `public` 중 하나여야
합니다. `unknown`, 공란, 승인되지 않은 데이터는 준비 단계에서 차단합니다.
