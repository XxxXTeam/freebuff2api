# freebuff2api

Codebuff Freebuff 的 OpenAI-compatible API

## 接口

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /healthz`

## 配置

复制 `.env.example` 为 `.env`，然后填写上游 token：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```dotenv
FREEBUFF_TOKEN=你的 Codebuff Bearer token
FREEBUFF_API_KEY=本地 OpenAI API key，可留空
FREEBUFF_AD_PROVIDERS=gravity,zeroclick
FREEBUFF_DEBUG=false
FREEBUFF_LOG_LEVEL=INFO
FREEBUFF_LOG_BODY_CHARS=2000
FREEBUFF_LOG_COLOR=true
```

当前内置 Freebuff 模型：

- `deepseek/deepseek-v4-flash`
- `deepseek/deepseek-v4-pro`
- `moonshotai/kimi-k2.6`
- `minimax/minimax-m2.7`

调试空返回或上游异常时：

```dotenv
FREEBUFF_DEBUG=true
FREEBUFF_LOG_LEVEL=DEBUG
FREEBUFF_LOG_BODY_CHARS=0
```

## 运行

```powershell
uv sync
uv run freebuff2api
```

或：

```powershell
python -m pip install -e .
python main.py
```

## 调用示例

```powershell
curl http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

流式：

```powershell
curl -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "写一个 Python 快排"}],
    "stream": true
  }'
```

## 感谢

> [FreeBuff](https://freebuff.com)
