# Qwen3 TTS 本地服务 API 集成指南

本文面向将 **Qwen3 TTS（本仓库 / streaming fork）** 作为后端能力接入桌面应用、本地 Web 或其它服务时的调用说明。默认假设推理在 **GPU（CUDA）** 上运行；若你自行改为 CPU，需单独评估性能与兼容性。

---

## 1. 文档说明

| 项目 | 说明 |
|------|------|
| **推荐 HTTP 接入** | 本仓库内置 **FastAPI 网关**（`qwen_tts/gateway`），提供常规 REST 与 **SSE 流式** 端点，**运行时不需要 Gradio，也不需要 `gradio_client`** |
| 可选：Gradio | 官方 CLI **`qwen-tts-demo`** 启动带网页 UI 的服务，协议为 **Gradio `/gradio_api/*`**，适合快速试听；程序化调用通常用 `gradio_client` 或继续用下方网关 |
| 本仓库特点 | Python 包内另有 **流式** API（`stream_generate_*`）；网关已将 **Base 模型** 的流式能力暴露为 **SSE**（见下文） |
| 适用读者 | 需要在本地应用中发起合成请求、处理音频与错误、并可选对接流式输出的开发者 |

---

## 2. 集成方式概览

| 方式 | 是否依赖 Gradio / `gradio_client` | 适用场景 |
|------|-------------------------------------|----------|
| **A. 内置 FastAPI 网关** | **否** | 任意语言的 HTTP 客户端（`curl`、`fetch`、移动端等），返回 `audio/wav` 或 SSE |
| **B. Gradio HTTP API** | 不强制，但协议为 Gradio；常用 `gradio_client` 省事 | 已启动 `qwen-tts-demo`、需要与官方 Demo 行为完全一致时 |
| **C. Python 进程内 API** | **否** | 同进程嵌入：`from qwen_tts import Qwen3TTSModel`，无 HTTP 开销 |

**结论：** 若你的目标是「本地起一个 API，用普通 HTTP 调」，请优先使用 **方式 A**，无需安装或使用 **`gradio_client`**。

---

## 3. 方式 A：内置 FastAPI 网关（推荐，无 Gradio）

模块位置：`qwen_tts/gateway`。使用 **FastAPI + Uvicorn**，与 Gradio **独立**。

### 3.1 环境变量

| 变量 | 含义 | 示例 |
|------|------|------|
| **`QWEN_TTS_MODEL_PATH`** | **必填**，Hugging Face 模型 ID 或本地目录 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `QWEN_TTS_DEVICE` | `device_map` | 默认 `cuda:0` |
| `QWEN_TTS_DTYPE` | 权重 dtype | 默认 `bfloat16`（可选 `float16` / `float32`） |
| `QWEN_TTS_FLASH_ATTN` | 是否 FlashAttention-2 | 默认 `1`（`0` 关闭） |
| `QWEN_TTS_CORS_ORIGINS` | CORS 来源 | 默认 `*`；多域名用英文逗号分隔 |

### 3.2 启动示例

在已激活虚拟环境、且已 `pip install -e .` 的前提下（示例端口 `8080`，与 Gradio 默认 `8000` 区分）：

**PowerShell：**

```powershell
$env:QWEN_TTS_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
$env:QWEN_TTS_DEVICE = "cuda:0"
uvicorn qwen_tts.gateway.app:app --host 127.0.0.1 --port 8080
```

若未设置 `QWEN_TTS_MODEL_PATH`，模块内的 `app` 可能为 `None`，Uvicorn 将无法加载；**务必先设置模型路径**。

启动后可用：

- `GET http://127.0.0.1:8080/health` — 健康检查  
- `GET http://127.0.0.1:8080/v1/meta` — 当前模型类型、设备、可选语种/说话人列表（若模型支持）  
- 浏览器打开 `http://127.0.0.1:8080/docs` — **Swagger UI**（OpenAPI），可直接试调

### 3.3 HTTP 端点一览

**说明：** 每个运行实例只加载 **一个** 模型；路由与 **模型类型** 必须一致（例如 Base 模型才能用 `voice_clone` 相关接口）。

| 方法 | 路径 | 内容类型 | 说明 |
|------|------|----------|------|
| GET | `/health` | JSON | 存活探测 |
| GET | `/v1/meta` | JSON | 模型与能力元数据 |
| POST | `/v1/tts/custom_voice` | JSON → **WAV** | **CustomVoice** 模型：预设说话人 + 可选指令 |
| POST | `/v1/tts/voice_design` | JSON → **WAV** | **VoiceDesign** 模型：音色描述指令 |
| POST | `/v1/tts/voice_clone` | `multipart/form-data` → **WAV** | **Base**：上传参考音频文件 + 表单字段 |
| POST | `/v1/tts/voice_clone/json` | JSON → **WAV** | **Base**：参考音频为 **Base64**（整段 WAV 等可读文件字节） |
| POST | `/v1/tts/voice_clone/prompt` | `multipart/form-data` → **WAV** | **Base**：上传 Gradio Demo 保存的 **`.pt` 音色文件** + 文本 |
| POST | `/v1/tts/voice_clone/stream` | JSON → **SSE** | **Base**：流式（`text/event-stream`），需 **streaming fork** 且实现含 `stream_generate_voice_clone` |

**JSON 公共字段（非流式）**

- 文本类接口 body 通常含：`text`、`language`（如 `"Auto"` / `"Chinese"`）等；可选嵌套 **`gen`** 对象，支持例如：`max_new_tokens`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`subtalker_*`、`non_streaming_mode`、`do_sample` 等（与模型能力一致，未传则用模型默认）。

**流式 SSE（`/v1/tts/voice_clone/stream`）**

- Body 为 JSON，含 `ref_audio_base64`、合成文本、可选 `emit_every_frames`、`decode_window_frames`、`overlap_samples` 等。  
- 响应为 **Server-Sent Events**：`event: chunk` 的 `data` 为 JSON，内含 `sample_rate` 与 `pcm_b64`（float32 PCM 小端字节再 Base64）；末尾 `event: done`。  
- 若当前安装非 streaming fork，可能返回 **501**。

### 3.4 调用示例（无需 `gradio_client`）

**健康检查：**

```bash
curl -s http://127.0.0.1:8080/health
```

**CustomVoice（JSON → 保存为 out.wav）：**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/tts/custom_voice" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"你好\",\"speaker\":\"Vivian\",\"language\":\"Auto\",\"instruct\":\"\"}" ^
  --output out.wav
```

**Base 语音克隆（multipart，参考音频 `ref.wav`）：**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/tts/voice_clone" ^
  -F "text=要合成的内容" ^
  -F "language=Chinese" ^
  -F "ref_text=参考音频对应的文本" ^
  -F "x_vector_only=false" ^
  -F "ref_audio=@ref.wav" ^
  --output clone.wav
```

**浏览器 / 前端：** 对上述 URL 使用 `fetch`、Axios、Kotlin `HttpURLConnection` 等即可；**不需要** 引入 Gradio 前端或 `gradio_client`。

---

## 4. 启动本地服务（Gradio，可选）

在已安装本仓库的环境中：

```powershell
Set-Location E:\Qwen3-TTS
.\.venv\Scripts\Activate.ps1
```

### 4.1 启动命令模板

```powershell
qwen-tts-demo <模型ID或本地路径> --device cuda:0 --ip 127.0.0.1 --port 8000
```

**常用 Hugging Face 模型 ID：**

| 模型 | 说明 |
|------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 基座：语音克隆 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 预设说话人 + 指令 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 文本描述音色 |

### 4.2 验证

- 页面：`http://127.0.0.1:8000`  
- API 元数据：`http://127.0.0.1:8000/gradio_api/info`

---

## 5. 方式 B：Gradio HTTP API 与 `gradio_client`（可选）

当你 **已经** 运行 `qwen-tts-demo` 且希望用 Gradio 自带协议调用时，可使用 **`gradio_client`（Python）** 或阅读 `/gradio_api/info` 自行对接。Gradio 6.x 的 API 前缀为 **`/gradio_api`**；队列开启时，**优先用 `gradio_client`** 处理会话与排队。

**最小示例（Python）：**

```python
from gradio_client import Client

client = Client("http://127.0.0.1:8000")
print(client.view_api())
```

Demo 中典型 `api_name`（以实际 `view_api()` 为准）：`/run_instruct`、`/run_voice_design`、`/run_voice_clone`、`/save_prompt`、`/load_prompt_and_gen`。

> **说明：** 若你已通过 **第 3 节网关** 接入业务，**不必** 再使用 `gradio_client`，除非你要同时驱动 Gradio 页面与同一后端（一般不需要）。

---

## 6. 方式 C：Python 进程内调用（无 HTTP）

适合 PyQt / 本地脚本与模型同进程：

```python
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

使用 `generate_voice_clone` / `generate_custom_voice` / `generate_voice_design`；streaming fork 另提供 `stream_generate_voice_clone` 等。

---

## 7. 网络、安全与稳定性

| 主题 | 建议 |
|------|------|
| 监听地址 | 网关与 Gradio 若仅本机使用，绑定 `127.0.0.1` |
| 鉴权 | 当前网关 **无内置鉴权**；公网暴露前请在外层加反向代理 + 认证 |
| Hugging Face | 首次拉模型需网络；内网请提前缓存模型 |
| 超时 | 长文本合成可能数十秒，HTTP 客户端超时建议 ≥ 120s 或异步任务 |
| SoX | 部分流程依赖系统 SoX，若报错请确认 `sox` 在 PATH 中 |

---

## 8. 常见问题（FAQ）

**Q：是否必须使用 `gradio_client` 才能调 API？**  
A：不是。使用本仓库 **FastAPI 网关（第 3 节）** 时，用任意 HTTP 客户端即可；只有对接 **Gradio Demo** 的 `/gradio_api` 时，才常用 `gradio_client` 简化协议。

**Q：网关和 Gradio 能同时开吗？**  
A：可以，请使用 **不同端口**（例如网关 `8080`，Gradio `8000`），各自加载模型会占 **双倍显存**；生产环境通常只保留一种。

**Q：`/gradio_api/info` 能访问，但直接 POST 失败？**  
A：Gradio 可能启用队列；请用 `gradio_client` 或改用 **第 3 节网关**。

**Q：流式合成如何用 HTTP？**  
A：对 Base 模型使用网关的 **`POST /v1/tts/voice_clone/stream`（SSE）**；非 streaming 上游则无此能力。

---

## 9. 参考

- FastAPI 文档：[https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- Gradio 文档：[https://www.gradio.app/docs](https://www.gradio.app/docs)
- `gradio_client`：[https://www.gradio.app/docs/python-client](https://www.gradio.app/docs/python-client)
- 上游模型说明：[https://github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- 本 streaming fork：`https://github.com/dffdeeq/Qwen3-TTS-streaming`

---

*网关实现见 `qwen_tts/gateway/`；Gradio 行为以运行中的 `/gradio_api/info` 为准。*
