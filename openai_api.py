import os
import openai
from dotenv import load_dotenv


# .env 파일에서 OpenAI API 키를 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response_from_openai(question, max_tokens=500):
    response = openai.Completion.create(
        model="text-davinci-003",  # 최신 모델로 업데이트
        prompt=question,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1.0,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()
