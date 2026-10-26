# 30분 Quick Practice

## AI가 일할 환경 만들기

- **회의록 오류 수정** 예제를 직접 실행해본다.
- 완성할 것은 Python 함수와 이를 수정·검증하는 하네스다.
- Codex CLI 또는 Claude Code 중 하나 사용을 권장한다.

## 수업 전 준비

- Python 3.10 이상과 Git을 준비하고, 사용할 CLI 설치 및 계정 로그인을 끝낸다.
- 이 예제는 LLM API를 호출하지 않는다.
- ZIP을 풀어 `학생용` 폴더를 연다. CLI의 작업 폴더는 반드시 `meeting-harness`로 한다.

### Codex 선택 시

macOS/Linux 설치 예시:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Windows 설치는 [공식 CLI 안내](https://developers.openai.com/codex/cli/)에서 Windows 탭의 현재 명령을 사용한다.

새 터미널에서 `codex --version`을 확인하고, 프로젝트 폴더에서 `codex`를 실행해 로그인한다.

### Claude Code 선택 시

macOS/Linux/WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell:

```powershell
irm https://claude.ai/install.ps1 | iex
```

새 터미널에서 `claude --version`을 확인하고, 프로젝트 폴더에서 `claude`를 실행해 로그인한다.
설치 문제는 [공식 Quickstart](https://code.claude.com/docs/en/quickstart)를 따른다.

### 시작 전 확인

터미널에서 `meeting-harness` 폴더로 이동한다. 아래 명령은 **셸에 입력**한다. 다음 안내의 프롬프트는 **CLI 대화창에 입력**한다.

```bash
python --version
git init
git add .
git commit -m "practice: starter"
```

`python` 명령이 없으면 macOS/Linux는 `python3`, Windows는 `py`를 사용한다. 이하 명령과 프롬프트의 `python`도 동일하게 바꾼다. 

Git 작성자 설정이 없으면 저장소 범위에서 본인 이름·이메일을 설정한 뒤 commit한다.

## 실습 문항


| 실습 문항                  | 학생이 남기는 결과              |
| ---------------------- | ----------------------- |
| ① 무엇을 확인하면 완료인가?       | 최초 테스트 기록, 실패 사례 1개 설명  |
| ② AI가 따라갈 규칙과 스킬 만들기   | 규칙 파일, 역할·흐름 문서, 검증 스킬  |
| ③ 작성→검토→완료 흐름으로 오류 고치기 | 수정 코드, 실제 검사 로그, 리뷰, 회고 |


전체 프롬프트는 `01_실습과_프롬프트.md`, 과제는 `02_프로젝트_과제.md`에 있다.

## 이론 복습

- 에이전트: CLI 안에서 코드와 도구를 다루는 AI. 이번에는 작성자와 검토자 역할을 순서대로 맡긴다.
- 스킬: 반복할 검사 절차를 `SKILL.md`에 저장하고 호출한다.
- 오케스트레이션: 학생이 작성→검토 순서를 실행하고, 실패 시 재시도 또는 중단을 결정한다.
- 하네스: 위 규칙·도구·테스트·기록이 함께 작동하는 환경이다.

규칙 문장의 수정 범위는 지침이며 실제 샌드박스가 아니다. CLI 기본 권한 확인을 유지하고 실제 변경 파일도 확인한다.



&nbsp;

# 하네스 엔지니어링 기초 실습

## 실습 소개

- **프로그램:** 회의록에서 `할 일:`로 시작하는 줄을 찾아 작업 목록으로 반환하는 Python 프로그램 ( 예: `할 일:자료 조사` → `["자료 조사"]`)
- **현재 오류:** 빈 입력에서 실행이 중단되고, 앞뒤 공백을 제대로 처리하지 못하며, 내용 없는 할 일도 목록에 포함된다.
- **수정할 내용:** `src/actions.py`에서 빈 입력은 `[]`로 반환하고, 공백을 정리하며, 빈 작업은 제외한다.
- **실습 목표:** AI에게 수정 규칙과 검사 스킬을 만들어 준 뒤, 작성 → 검사 → 검토 순서로 오류를 고친다. 테스트 8개 통과와 예제 출력으로 결과를 확인한다.

  

## ① 실패를 관찰하고 완료 기준 정하기

### 문항

“회의록 앱을 고쳐줘” 대신 무엇을 확인해야 완료인지 설명해보세요.

현재 코드에서 실패한 입력 1개를 찾고 기대값과 실제 결과를 비교해보세요.

### 실행 프롬프트

```text
현재 작업 폴더가 meeting-harness인지 확인해 줘.
requirements.md, src/actions.py, tests/test_actions.py, demo.py를 읽어 줘.
아직 앱 코드나 테스트를 수정하지 마.
python -m unittest discover -s tests -v 를 실제 실행하고,
명령, 종료 코드, 원본 출력을 workspace/baseline.txt에 저장해 줘.
현재 환경에서 python 대신 python3 또는 py가 필요하면 그 명령을 사용하고 기록해 줘.
실패 사례 1개를 입력 / 기대 결과 / 실제 결과 / 요구사항 번호로 설명해 줘.
검사를 실행할 수 없다면 원인을 기록하고 멈춰. 실행했다고 추측하지 마.
```

### 예상 결과·산출물

- `workspace/baseline.txt`: 8개 중 **4개 통과, 2개 실패, 2개 오류**.
- 예: `""` → 기대 `[]`, 실제 `ValueError` 발생.
- 코드와 테스트에는 아직 변경이 없다.

**확인 질문:** AI의 “수정 가능해요”와 테스트 실행 기록 중 무엇이 현재 상태의 증거일까요?

## ② 규칙과 검증 스킬 만들기

### 문항

누가 어떤 파일을 수정할지, 어떤 검사로 완료를 판단할지 정하라. 같은 검사를 다음 작업에서도 실행할 수 있도록 스킬로 저장하세요

Codex:

```text
나는 Codex CLI를 사용해. 규칙은 AGENTS.md, 스킬은 .agents/skills/check-meeting/SKILL.md에 만들어 줘.
```

Claude Code:

```text
나는 Claude Code를 사용해. 규칙은 CLAUDE.md, 스킬은 .claude/skills/check-meeting/SKILL.md에 만들어 줘.
```

### 프롬프트

```text
requirements.md와 현재 파일 구조를 근거로 작은 개발 하네스를 실제 파일로 작성해 줘.
이번 단계에서는 src/, tests/, demo.py, requirements.md를 수정하지 마.
1. 선택한 규칙 파일: 목적, 수정 가능 범위, 테스트 명령, 완료 기준을 짧게 적어 줘.
   앱 수정은 src/actions.py만 허용한다. tests/, 요구사항, demo.py는 바꾸지 않는다.
   외부 패키지 설치, API 연결, 기능 추가는 하지 않는다.
2. workflow.md: 작성자와 검토자의 입력/출력/수정 범위를 적어 줘.
   작성자는 코드와 workspace/change.md, 검토자는 workspace/review.md를 남긴다.
   순서는 작성 → 검사·검토 → 학생 최종 확인이다.
   검토 실패 시 작성 단계로 돌아가며 추가 수정은 최대 2회, 계속 실패하면 중단한다.
   독립 서브에이전트는 만들지 않는다.
3. 선택한 경로에 check-meeting 스킬을 만들어 줘.
   SKILL.md 맨 앞 YAML에 name: check-meeting 과 description을 넣어 줘.
   description에는 '회의록 추출기 변경 후 요구사항과 테스트를 검증할 때 사용'을 담아 줘.
   본문: 요구사항 읽기 → 실제 테스트 실행 → demo.py 실행 → 결과 기록 → 통과/실패/미실행 판정.
   workspace/checks.txt에 실제 명령, 종료 코드, 출력과 실패 원인을 기록한다.
   반복 실행 시 기존 기록을 남기고 실행 번호를 붙여 추가한다.
   이 스킬은 앱 코드나 테스트를 고치지 않는다. 실행 실패나 테스트 실패를 성공으로 쓰지 않는다.
명령 이름은 앞 단계에서 동작한 Python 실행 명령으로 통일해 줘.
만든 파일 경로와 핵심 규칙을 짧게 설명해 줘.
```

### 예상 결과·산출물


| Codex                                   | Claude Code                             | 역할               |
| --------------------------------------- | --------------------------------------- | ---------------- |
| `AGENTS.md`                             | `CLAUDE.md`                             | 계속 참고할 프로젝트 규칙   |
| `.agents/skills/check-meeting/SKILL.md` | `.claude/skills/check-meeting/SKILL.md` | 필요할 때 실행할 검사 절차  |
| `workflow.md`                           | `workflow.md`                           | 작성·검토 책임과 재시도 흐름 |


스킬의 시작은 아래 형태여야 한다. 코드 펜스까지 파일에 감싸 넣지 않는다.

```yaml
---
name: check-meeting
description: 회의록 추출기 변경 후 요구사항과 테스트를 검증할 때 사용한다.
---
```

**확인 질문:** “테스트 실행 방법”은 규칙·역할·스킬 중 어디에 자세히 적는가? 왜 매번 긴 프롬프트를 다시 붙이지 않아도 될까요?

## ③ 작성 → 검토 → 완료 실행하기

### 문항

규칙에 맞춰 오류를 수정하고, 스킬을 호출해 실제로 확인해보세요. AI의 설명 대신 실행 증거로 통과 여부를 결정하세요.

### 작성자에게 요청

```text
프로젝트 규칙 파일과 workflow.md, requirements.md를 읽고 작성자 역할로 작업해 줘.
src/actions.py만 수정해서 요구사항을 만족시켜 줘. tests/와 demo.py는 바꾸지 마.
수정 전 문제, 수정 내용, 변경 파일을 workspace/change.md에 적어 줘.
검사를 했다면 실제 실행 여부와 결과를 기록하고, 하지 않았다면 미실행이라고 적어 줘.
작성 단계가 끝나면 멈추고 검토 단계로 넘겨 줘.
```

### 검증 스킬 호출

새 규칙을 확실히 읽도록 CLI를 종료하고 같은 폴더에서 다시 실행한다. 다음 중 하나를 입력한다.

Codex 대화창:

```text
$check-meeting
```

Claude Code 대화창:

```text
/check-meeting
```

스킬이 보이지 않으면 경로·YAML을 확인하고 CLI를 재시작한다.

Codex는 `/skills`에서도 찾을 수 있다. 그래도 인식되지 않으면 SKILL.md를 직접 읽어 절차대로 실행하도록 요청한다.

### 검토자에게 요청

```text
이제 검토자 역할이다. 규칙 파일, workflow.md, requirements.md,
src/actions.py, tests/test_actions.py, workspace/change.md, workspace/checks.txt를 읽어 줘.
git diff -- src/actions.py tests demo.py requirements.md 와 git status --short를 확인해 줘.
앱 코드나 테스트를 수정하지 마.
요구사항 충족, 변경 범위 준수, 실제 테스트와 demo 실행 증거를 검토해서
workspace/review.md에 통과/수정 필요/판정 불가 중 하나와 근거를 적어 줘.
기록에 없는 실행을 상상하지 마. 테스트 통과가 모든 입력의 정확성을 보장한다고 쓰지 마.
수정이 필요하면 작성자에게 넘길 구체적 항목을 적고 멈춰 줘.
```

검토가 실패하면 학생이 기존 프롬프트에 수정 항목을 붙여 다시 요청하고 이후 프롬프트를 반복한다.

**추가 수정 최대 2회** 후에도 실패하면 실패 기록을 남기고 중단한다. 재시도 제한은 학생이 관리한다.

### 최종 확인·제출

터미널에서 직접 다시 실행한다.

```bash
python -m unittest discover -s tests -v
python demo.py
git diff -- src/actions.py tests demo.py requirements.md
git status --short
```

예상 테스트 출력의 핵심:

```text
Ran 8 tests in ...s
OK
```

예상 demo 출력:

```text
빈 회의록: []
할 일 목록: ['자료 조사', '슬라이드 작성']
```

AI 생성 문서와 `workspace/`는 새 파일로 표시될 수 있다. 기존 파일 변경은 `src/actions.py`만 있어야 한다. `git diff`는 새 파일 내용을 보여주지 않으므로 규칙·스킬·기록 파일도 직접 연다.

`03_제출_양식.md`를 프로젝트의 `workspace/submission.md`로 복사하여 채운다.

제출: 수정 코드 + 규칙 + 스킬 + workflow.md + workspace/ 기록.

GitHub 제출 시 /harness 폴더를 만든 후 산출물을 위치시키고 제출한다.



**확인 질문:** 이번에 만든 하네스에서 규칙·절차·검사·기록을 각각 한 파일 또는 행동에 연결해 보세요. 사람은 언제 판단할까요?