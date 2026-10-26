def extract_actions(text):
    """회의록의 '할 일:' 줄에서 작업 내용을 추출한다."""
    if not text.strip():
        raise ValueError("회의록이 비어 있습니다")
    return [line.split(":", 1)[1] for line in text.splitlines()
            if line.startswith("할 일:")]
