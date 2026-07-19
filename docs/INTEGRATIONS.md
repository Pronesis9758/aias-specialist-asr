# GitHub, Hugging Face, Colab integration

## GitHub

대상 저장소는 `Pronesis9758/aias-specialist-asr`입니다. 실제 음성 파일은 제외하고
코드·config·작은 합성 fixture만 게시합니다. 공개 음성은 Colab 실행 시 Drive로 내려받으며
Git에는 포함하지 않습니다.

현재 Codex의 GitHub 앱에서는 이 저장소가 404로 조회됩니다. 비공개 저장소라면 GitHub 앱을
`Pronesis9758` 계정 또는 해당 저장소에 설치하고, 로컬에는 GitHub CLI를 설치·인증해야
커밋·push·PR 자동화가 가능합니다.

연결 후 확인:

```powershell
git remote -v
git status --short --branch
```

저장소에 push되면 `.github/workflows/ci.yml`이 fixture 파이프라인 테스트와 정적 분석을
자동으로 실행합니다. CI는 모델이나 실제 음성 데이터를 다운로드하지 않습니다.

## Hugging Face

공개 모델과 `kresnik/zeroth_korean`은 토큰 없이 받을 수 있습니다. gated/private 모델 또는
private dataset을 쓸 때만
write 권한이 없는 최소 read token을 `HF_TOKEN` 환경변수에 저장합니다. 토큰은 Git에 커밋하지
않습니다.

```powershell
uv run hf auth whoami
uv run aias model-lock --config configs/local_baseline.yaml
uv run aias download-model --config configs/local_baseline.yaml
```

## Google Colab

Colab은 개발환경이 아니라 GPU 실행기로 사용합니다.

1. GitHub에서 코드를 clone/pull한다.
2. 공개 데이터셋의 고정 리비전에서 64개 샘플을 Drive로 스트리밍한다.
3. Baseline 모델과 학습 모델의 리비전을 commit SHA로 고정한다.
4. checkpoint를 주기적으로 Drive에 저장하고 `resume_from_checkpoint=auto`를 사용한다.
5. 결과, Word 보고서, backdata SQLite를 Drive에 저장한다.

사용자가 직접 해야 하는 단계:

- Colab에서 Google 계정의 Drive 마운트 승인
- 비공개 GitHub 저장소라면 Colab Secrets에 최소 읽기 권한의 `GITHUB_TOKEN` 등록
- GPU 런타임 선택 및 필요 시 요금 결제
- 실제 사내 데이터의 Colab 사용 승인
