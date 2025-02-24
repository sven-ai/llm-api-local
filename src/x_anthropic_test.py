import os, anthropic

if not os.environ.get("ANTHROPIC_API_KEY"):
    print('ERR. `ANTHROPIC_API_KEY` env var is not set.')

res = anthropic.Anthropic().messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)
print(res)