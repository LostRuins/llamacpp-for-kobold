import openai
openai.api_key = "sk-test"
openai.api_base = "http://localhost:5001/api/extra/oai/v1"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
  ],
  max_tokens=250,
)
print("whole response: \n")
print(completion)
print("\nMessage content:\n")
print(completion.choices[0].message)