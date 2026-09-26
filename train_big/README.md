# Train bản lớn đa nguồn (miễn phí qua GitHub Actions)

## Vì sao đa nguồn?
Trước đây chỉ học từ 1 nền tảng (Chatbot Arena) → dễ "học vẹt" phong cách của nền tảng đó.
Bản này trộn **3 nguồn khác nhau**:

| Nguồn | Loại | Thời gian | Đặc biệt |
|---|---|---|---|
| arena-human-preference-140k | battle chat | 2024–05/2025 | Prompt đa dạng, quốc tế |
| arena-expert-5k | battle chat (chuyên gia chấm) | 11/2025 | GPT-5, Opus 4.1, GLM 4.5... |
| WildChat-1M | hội thoại người dùng thật | 2024 | Nền tảng khác hoàn toàn → tăng khách quan |

Quy tắc cân bằng: mỗi class tối đa 2.200 mẫu, **một nguồn tối đa 60%** mẫu của class.

## Chạy miễn phí trên GitHub Actions
Repo **public** → minutes không giới hạn, máy `ubuntu-latest` (4 vCPU / 16GB RAM).

1. Tạo repo public, đẩy code này lên (kèm `.github/workflows/train.yml`)
2. Thêm secret `HF_TOKEN` (token HuggingFace có quyền ghi repo)
3. Tab **Actions → Train AI Detector → Run workflow**
4. Xong: model nằm ở https://huggingface.co/chiminh652010/ai-detector-v2
   (tải `detector.joblib` về thay vào `model/detector.joblib` là xong)

## Cấu hình bản lớn (cần ~10GB RAM — sandbox 2GB không nổi)
- MAXC 2.200 mẫu/class (~55+ class)
- Vocab học từ 12.000 mẫu: word 60.000 (1–3 gram) + char 24.000 (2–4 gram)
- WORD_LEN 2500 / CHAR_LEN 1200
- LR C=2.0, struct 28 đặc trưng

Kỳ vọng: top-1 ~33–36% với ~57 class (so 29.6% bản sandbox),
tổng thời gian ~1.5–2 giờ.
