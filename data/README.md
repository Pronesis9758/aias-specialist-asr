# Data contract

- `sample/`: 공개 가능한 합성 fixture. CI와 smoke test에서만 사용합니다.
- `domain_terms/`: 현업 검수를 거친 canonical 제조 용어와 ASR 오인식 alias입니다.
- `raw_audio/`: 공개 가능 음성만 임시로 둘 수 있으며 기본적으로 Git에서 제외됩니다.
- `private/`: 실제 음성/전사 manifest 위치. 전체 디렉터리가 Git에서 제외됩니다.

실제 데이터 manifest의 `consent_status`는 최소 `approved`, `synthetic`, `public` 중 하나여야
합니다. `unknown`, 공란, 승인되지 않은 데이터는 준비 단계에서 차단합니다.

실제 제조 데이터는 `templates/manufacturing_manifest_template.csv`의 열을 사용합니다.
`governance.mode=strict_private`에서는 다음을 추가로 검증합니다.

- 완료된 `data_approval.yaml`과 일치하는 `approval_id`
- `deidentified=true`
- `label_review_status=reviewed`와 검수자
- 화자가 Train/Validation/Test 중 하나에만 포함되는지

양식은 `templates/data_approval_template.yaml`과
`templates/human_review_signoff_template.yaml`에 있습니다. 실제 값이 들어간 파일은
`data/private/` 또는 승인된 비공개 Drive에 두고 Git에 커밋하지 않습니다.
