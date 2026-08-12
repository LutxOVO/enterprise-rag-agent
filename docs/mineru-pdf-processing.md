# MinerU 大 PDF 处理说明

## 为什么需要拆分

MinerU 接口对单次解析的 PDF 页数有限制。当前项目默认把这个限制配置为 200 页：

```env
MINERU_MAX_PAGES_PER_REQUEST=200
```

如果上传的 PDF 有 320 页，直接把整个文件交给 MinerU 会得到类似下面的错误：

```text
number of pages exceeds limit (200 pages), please split the file and try again
```

项目现在会自动处理这个场景，不需要用户手工切 PDF。

## 代码调用链

```text
POST /api/documents/upload
        |
        v
DocumentService._extract_with_mineru_api()
        |
        v
MinerUClient.parse_file_to_markdown()
        |
        +-- PDF <= 配置上限: _parse_single_file()
        |
        +-- PDF > 配置上限:
              pypdf 读取页数
              -> 生成 part_001.pdf、part_002.pdf ...
              -> 每个分片调用 _parse_single_file()
              -> 按顺序合并 Markdown
              -> 删除临时分片 PDF
        |
        v
Markdown 切分 -> Embedding -> Chroma -> PostgreSQL
```

关键文件：

- `app/core/config.py`：定义 `mineru_max_pages_per_request` 配置。
- `app/services/mineru_client.py`：负责页数判断、PDF 拆分、MinerU 请求和 Markdown 合并。
- `app/services/document_service.py`：把 MinerU 返回的 Markdown 交给切分和向量入库流程。

## 320 页文件会发生什么

假设上传文件名是 `AI-Agents-in-Depth.pdf`，它有 320 页：

1. `pypdf` 发现总页数为 320。
2. 项目生成两个临时文件：第一片 1-200 页，第二片 201-320 页。
3. 第一片输出写入 `data/mineru_output/<文档名>/part_001/`。
4. 第二片输出写入 `data/mineru_output/<文档名>/part_002/`。
5. 两次解析得到的 Markdown 按 `part_001`、`part_002` 顺序拼接。
6. 临时目录 `_split_pdfs` 被删除；原始 PDF 和分片解析结果保留。

因此，这个功能解决的是“单次请求超过页数限制”，不是减少 MinerU 计费或额度消耗。320 页仍然需要两次解析请求。

## 为什么使用 pypdf

这里的 `pypdf` 只负责两件事：读取页数和写出 PDF 分片。真正的复杂版面识别、OCR、表格和公式解析仍然由 MinerU 完成。

这样做的好处是：

- 不需要在本地下载或加载 MinerU 模型。
- 保留 MinerU 对扫描 PDF、表格和复杂版面的处理能力。
- 临时拆分文件不会一次性把整个 PDF 转成大块字符串。

## 配置和运行

确认 `.env` 中至少有：

```env
PDF_PARSER="mineru_api"
MINERU_API_TOKEN="你的 MinerU API Token"
MINERU_MAX_PAGES_PER_REQUEST=200
```

启动服务：

```powershell
./scripts/run_local.ps1 -Reload
```

上传原始 PDF：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload" `
  -F "file=@C:\Users\15963\Downloads\AI-Agents-in-Depth-zh-CN.pdf"
```

批量上传也会复用同一套拆分逻辑：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload-batch" `
  -F "files=@C:\Users\15963\Downloads\AI-Agents-in-Depth-zh-CN.pdf"
```

## 失败和重试

- MinerU 解析某个分片失败时，接口会返回失败；原始上传 PDF 不会删除。
- 本次请求生成的临时拆分 PDF 会被清理，避免下次重试误用旧分片。
- 批量上传可以通过批次明细接口重试失败项；重试时会复用原始文件。
- 已经成功生成的 `part_001` 等输出目录会保留，便于排查，但重新解析时会覆盖对应的结果文件。
- 把上限设置得更小会增加 API 请求次数；只有在服务端限制更严格或希望降低单次请求压力时才这样配置。

## 本地测试

测试使用 `pypdf` 生成 2 页、5 页等临时 PDF，并 mock `_parse_single_file()`，不会访问真实 MinerU：

```powershell
uv run pytest tests/test_mineru_client.py -q
```

测试覆盖：

- 未超过上限的 PDF 只调用一次。
- 5 页 PDF 在上限为 2 时拆成 2、2、1 页。
- Markdown 合并顺序正确。
- 成功和失败时临时分片都会被删除。
