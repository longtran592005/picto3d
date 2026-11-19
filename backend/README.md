# Picto3D Backend — Image → GLB

FastAPI service biến 1 hoặc nhiều ảnh của cùng một vật thể thành mô hình 3D `.glb`
bằng **TripoSR**, có tách nền (rembg), căn khung 80%, tối ưu lưới 10k–100k mặt
(mặc định 40k) và tự giải phóng VRAM sau mỗi lần dựng.

## Kiến trúc

```
Upload ảnh
   │
   ▼
decode + EXIF + giới hạn kích thước
   │
   ▼
rembg (BiRefNet/ISNet) → alpha cutout → chấm điểm từng góc nhìn
   │
   ▼
căn giữa, foreground ratio 0.8, nền xám 127, 512×512
   │
   ▼
TripoSR (marching cubes 256) → mesh + vertex color
   │
   ▼
làm sạch → chuẩn hoá Y-up → decimate 40k → Taubin smooth → căn feet về Y=0
   │
   ▼
export GLB → /files/picto3d_<job>.glb
```

## Chạy trên Google Colab (T4)

1. Mở [Colab](https://colab.research.google.com), chọn
   `Runtime → Change runtime type → T4 GPU`.
2. Mở `backend/colab/Picto3D_Backend.ipynb`. Notebook tự clone
   `https://github.com/longtran592005/picto3d.git` về `/content/picto3d`
   (hoặc `git pull` nếu đã có). Muốn dùng fork khác thì sửa `REPO_URL` trong cell 1.
3. Chạy lần lượt các cell.
4. Cell cuối in ra:

```
╔══════════════════════════════════════════════════╗
║ PICTO3D API ĐÃ SẴN SÀNG                          ║
║ API URL : https://xxx.trycloudflare.com          ║
╚══════════════════════════════════════════════════╝
```

Dán API URL đó vào ô **Backend API** của web studio.

> Lần chạy đầu sẽ tải model TripoSR (~1.5 GB) + model tách nền. Giữ tab Colab mở.

## Chạy trên server Linux GPU

```bash
# 1) PyTorch khớp CUDA (ví dụ CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Dependencies
cd backend
pip install -r requirements.txt

# 3) TripoSR + marching cubes
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR
pip install git+https://github.com/tatsy/torchmcubes.git
export PICTO3D_TRIPOSR_PATH=/opt/TripoSR

# 4) Chạy
python colab_launch.py --port 8000              # kèm cloudflared
python colab_launch.py --port 8000 --no-tunnel  # chỉ local
# hoặc: uvicorn app:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*"
```

Mở http://localhost:8000/docs để xem Swagger UI.

### ngrok (thay cho cloudflared)

```bash
pip install pyngrok
export NGROK_AUTHTOKEN=...        # nếu tài khoản yêu cầu
python colab_launch.py --tunnel ngrok
```

Cloudflared thủ công:

```bash
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

## API

### `POST /api/reconstruct`

`multipart/form-data`:

| Field | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `files` | file[] | — | 1–8 ảnh (PNG/JPG/WEBP), tối đa 25 MB/ảnh |
| `foreground_ratio` | float | `0.8` | tỉ lệ vật thể trong khung vuông |
| `mc_resolution` | int | `256` | độ phân giải marching cubes (128–384) |
| `target_faces` | int | `40000` | số mặt sau decimate (10k–100k) |
| `remove_background` | bool | `true` | tắt nếu ảnh đã tách nền sẵn |
| `wait` (query) | bool | `true` | `false` → trả `job_id` để polling |

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/reconstruct \
  -F "files=@front.png" \
  -F "files=@side.png" \
  -F "foreground_ratio=0.8" \
  -F "target_faces=40000"
```

Kết quả:

```json
{
  "job_id": "8f3a1c2b9d10",
  "status": "completed",
  "glb_url": "https://xxx.trycloudflare.com/files/picto3d_8f3a1c2b9d10.glb",
  "download_url": "https://xxx.trycloudflare.com/api/download/8f3a1c2b9d10",
  "file_name": "picto3d_8f3a1c2b9d10.glb",
  "elapsed_seconds": 14.82,
  "preprocess_seconds": 2.1,
  "inference_seconds": 9.4,
  "mesh_seconds": 3.3,
  "vertices": 20482,
  "faces": 40000,
  "texture": "vertex_color",
  "input_views": 2,
  "selected_view": 1,
  "device": "cuda"
}
```

### Các endpoint khác

| Endpoint | Mô tả |
|---|---|
| `GET /api/health` | trạng thái GPU, model, queue |
| `GET /api/jobs/{id}` | tiến trình: `preprocess → inference → mesh → export` |
| `GET /api/download/{id}` | tải GLB kèm `Content-Disposition` |
| `GET /files/{name}` | file tĩnh |
| `GET /docs` | Swagger |

Job bất đồng bộ:

```bash
curl -X POST "http://localhost:8000/api/reconstruct?wait=false" -F "files=@front.png"
# → {"job_id":"...","status":"queued",...}
curl http://localhost:8000/api/jobs/<job_id>
```

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `PICTO3D_DEVICE` | `auto` | `cuda`, `cuda:0`, `cpu` |
| `PICTO3D_CHUNK_SIZE` | `8192` | giảm khi OOM |
| `PICTO3D_MC_RESOLUTION` | `256` | chất lượng lưới mặc định |
| `PICTO3D_TARGET_FACES` | `40000` | ngân sách mặt cho web |
| `PICTO3D_FOREGROUND_RATIO` | `0.8` | tỉ lệ vật thể |
| `PICTO3D_SEG_MODEL` | `isnet-general-use` | đổi sang `birefnet-general` để nét hơn (tải ~900 MB) |
| `PICTO3D_TRIPOSR_PATH` | `/content/TripoSR` | thư mục chứa `tsr/` |
| `PICTO3D_OUTPUT_DIR` | `backend/outputs` | nơi lưu GLB |
| `PICTO3D_FILE_TTL` | `3600` | giây trước khi xoá file |
| `PICTO3D_MAX_UPLOAD_MB` | `25` | giới hạn mỗi ảnh |
| `PICTO3D_MAX_IMAGES` | `8` | số ảnh mỗi request |
| `PICTO3D_PUBLIC_URL` | _(trống)_ | override base URL khi chạy sau reverse proxy |
| `PICTO3D_PRELOAD` | `true` | nạp model nền lúc khởi động |
| `PICTO3D_WARMUP` | `true` | chạy warm-up 1 ảnh xám sau khi nạp model |
| `PICTO3D_MC_CPU_FALLBACK` | `true` | dùng PyMCubes nếu thiếu torchmcubes |

## Đa góc nhìn

TripoSR là mô hình **single-image**. Backend nhận nhiều ảnh, tách nền và chấm điểm
từng góc (coverage, độ nét, độ cân giữa) rồi chọn góc tốt nhất — ưu tiên ảnh đầu tiên
trừ khi ảnh khác rõ ràng tốt hơn. `input_views` và `selected_view` trong response cho
biết backend đã dùng gì. Muốn fusion thật sự nhiều góc, có thể thay engine bằng
InstantMesh qua interface `ReconstructionEngine` (`picto3d/engines/base.py`).

## Hiệu năng tham khảo (Tesla T4)

| Bước | Thời gian |
|---|---|
| tách nền + căn khung | 1–3 s / ảnh |
| TripoSR forward + mesh 256 | 6–12 s |
| decimate + smooth + export | 2–4 s |
| **Tổng** | **~12–20 s** |

VRAM đỉnh ~6–8 GB. `torch.cuda.empty_cache()` + `gc.collect()` chạy trong `finally`
sau mỗi job.

## Troubleshooting

- **`Không tìm thấy mã nguồn TripoSR`**: clone repo và đặt `PICTO3D_TRIPOSR_PATH`.
- **`torchmcubes` build lỗi**: `pip install PyMCubes`; backend tự shim (chậm hơn).
- **CUDA OOM**: giảm `mc_resolution` (192), `target_faces` (30000),
  `PICTO3D_CHUNK_SIZE=4096`; khởi động lại runtime Colab.
- **Lỗi transformers/huggingface_hub**: giữ đúng cặp
  `transformers==4.35.0` + `huggingface-hub==0.25.2`.
- **Web báo CORS**: kiểm tra URL không có dấu `/` thừa và server trả
  `Access-Control-Allow-Origin: *` (đã bật mặc định).
- **Tunnel đổi URL sau khi restart**: dán URL mới vào web studio.

## Cấu trúc

```
backend/
├── app.py                  # FastAPI app + routes
├── colab_launch.py         # uvicorn + cloudflared/ngrok
├── requirements*.txt
├── colab/Picto3D_Backend.ipynb
└── picto3d/
    ├── config.py           # env settings
    ├── preprocessing.py    # rembg, framing, scoring
    ├── engines/            # TripoSR wrapper (+ interface)
    ├── mesh_ops.py         # clean, decimate, smooth, GLB
    ├── pipeline.py         # orchestration + progress
    ├── jobs.py             # job store + TTL
    ├── schemas.py
    ├── runtime.py          # timers, VRAM release
    └── errors.py
```

## Lưu ý bảo mật

Quick tunnel và CORS `*` nghĩa là **ai có URL cũng gọi được GPU của bạn**. Chỉ chia sẻ
URL trong lúc demo, tắt Colab khi xong. Nếu triển khai lâu dài, thêm API key/reverse
proxy và giới hạn `allow_origins`.
