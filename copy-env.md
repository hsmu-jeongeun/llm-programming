### 아나콘다 가상환경 복사 방법

본인 가상환경에서 실습 진행했을 때, 코드 실행이 안되는 경우 의존성 문제 해결을 위해 진행

1. `conda-llm-env.yaml` 다운로드
2. 터미널 연 후 다운로드한 위치로 이동
   예시)
   ```shell
   cd C:\Users\{user_name}\Downloads
   ```
3. 다음 명령어 실행
    ```shell
   cd conda env create -f {원하는 가상환경 이름} conda-llm-env.yaml
   ```
4. 정상적으로 생성되었는지 활성화해보기
    ```shell
   conda activate {신규 생성 가상환경 이름}
   ```