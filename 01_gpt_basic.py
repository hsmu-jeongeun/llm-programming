from openai import OpenAI

api_key = "your_api_key_here"
client = OpenAI(api_key = api_key) 

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승 팀은 어디야?"}
    ]
)

print(response)
print(response.choices[0].message.content)