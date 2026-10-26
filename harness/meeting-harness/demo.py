from src.actions import extract_actions

sample = "회의 주제: 발표 준비\n  할 일:  자료 조사  \n할 일:슬라이드 작성\n할 일:   "
print("빈 회의록:", extract_actions(""))
print("할 일 목록:", extract_actions(sample))
