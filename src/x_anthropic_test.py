import os, anthropic

if not os.environ.get("ANTHROPIC_API_KEY"):
    print('ERR. `ANTHROPIC_API_KEY` env var is not set.')

res = anthropic.Anthropic().messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 16000
    },
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)
print(res)