from fastapi import FastAPI, UploadFile, File, HTTPException, Form,Request,Form
from fastapi.responses import JSONResponse, FileResponse,HTMLResponse
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import uuid
import json
app = FastAPI(title="Veri Seti İşleme ve Dönüştürme API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Geçici dosyaların saklanacağı klasör
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

def save_temp_file(contents: bytes, filename: str) -> str:
    path = os.path.join(TEMP_FOLDER, filename)
    with open(path, "wb") as f:
        f.write(contents)
    return path

@app.get("/")
def read_root():
    return {"message": "FastAPI çalışıyor!"}


@app.post("/upload", response_class=HTMLResponse)
async def upload_dataset(dataset: UploadFile = File(...)):
    if not dataset.filename.endswith('.csv'):
        return HTMLResponse("<p class='text-red-600'>❌ Sadece CSV dosyaları destekleniyor.</p>", status_code=400)

    contents = await dataset.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        return HTMLResponse(f"<p class='text-red-600'>❌ Dosya okunamadı: {str(e)}</p>", status_code=400)

    file_id = str(uuid.uuid4())
    file_path = save_temp_file(contents, f"{file_id}_{dataset.filename}")

    # Dataset bilgileri
    shape_info = f"{df.shape[0]} satır × {df.shape[1]} kolon"
    column_info = "".join([
        f"<tr><td class='border px-2 py-1'>{col}</td><td class='border px-2 py-1'>{str(dtype)}</td></tr>"
        for col, dtype in df.dtypes.items()
    ])

    html = f"""
    <div class="border rounded p-4 bg-gray-50">
      <h3 class="text-lg font-semibold mb-2">📄 CSV Özeti</h3>
      <p><strong>Dosya Adı:</strong> {dataset.filename}</p>
      <p><strong>Boyut:</strong> {shape_info}</p>

      <div class="overflow-x-auto mt-4">
        <table class="table-auto border text-sm w-full">
          <thead class="bg-gray-200">
            <tr>
              <th class="border px-2 py-1 text-left">Kolon</th>
              <th class="border px-2 py-1 text-left">Veri Tipi</th>
            </tr>
          </thead>
          <tbody>
            {column_info}
          </tbody>
        </table>
      </div>

      <form hx-post="http://localhost:8000/process"
            hx-target="#step2-container"
            hx-swap="innerHTML"
            class="mt-6 space-y-2">
        <input type="hidden" name="file_id" value="{file_id}" />
        <input type="hidden" name="filename" value="{dataset.filename}" />
        <button type="submit" class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
          İşle ve Devam Et
        </button>
      </form>
    </div>
    """

    return HTMLResponse(content=html)


from fastapi.responses import HTMLResponse

@app.post("/process", response_class=HTMLResponse)
async def process_dataset(file_id: str = Form(...), filename: str = Form(...)):
    file_path = os.path.join(TEMP_FOLDER, f"{file_id}_{filename}")
    if not os.path.exists(file_path):
        return HTMLResponse("<p class='text-red-600'>❌ Dosya bulunamadı.</p>", status_code=404)

    try:
        # 1️⃣ CSV işle
        raw_path = os.path.join("temp_files", f"{file_id}_{filename}")
        df = pd.read_csv(raw_path)

        # 2️⃣ Boolean dönüşümü
        for col in df.columns:
            if df[col].dtype == object:
                vals = df[col].dropna().unique()
                if set(vals).issubset({"True", "False"}):
                    df[col] = df[col].map({"True": 1, "False": 0})

        # 3️⃣ Categorical sütunları encode et (< 50 unique ise)
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() < 50:
                df[col] = pd.Categorical(df[col]).codes

        # 4️⃣ Eksik veri doldur ve tekrarları sil
        df.ffill( inplace=True)
        df.drop_duplicates(inplace=True)

        # 5️⃣ İşlenmiş dosyayı kaydet
        processed_filename = f"processed_{file_id}_{filename}"
        processed_path = os.path.join("temp_files", processed_filename)
        df.to_csv(processed_path, index=False)
    except Exception as e:
        return HTMLResponse(f"<p class='text-red-600'>❌ İşleme hatası: {str(e)}</p>", status_code=400)

    try:
        # 2️⃣ frontend klasöründen Front.html oku
        front_path = os.path.abspath(
            "C:/Users/HP/Desktop/GP2/html/GP2/Front.html"
                )


        with open(front_path, "r", encoding="utf-8") as f:
            html = f.read()

        html = html.replace("{{ file_id }}", file_id)
        html = html.replace("{{ filename }}", processed_filename)

        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(f"<p class='text-red-600'>❌ Front.html okunamadı: {str(e)}</p>", status_code=500)




@app.post("/train")
async def train_model(
    request: Request,
    file_id: str = Form(...),
    filename: str = Form(...),
    loss_function: str = Form(...),
    optimizer: str = Form(...),
    learning_rate: float = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    validation_split: int = Form(...),
    loss_monitor_freq: int = Form(...),
    shuffle: str = Form(...),
    enable_early_stopping: Optional[bool] = Form(False),
    early_stopping_patience: Optional[int] = Form(5),
    early_stopping_delta: Optional[float] = Form(0.001),
    enable_lr_adapter: Optional[bool] = Form(False),
    lr_monitor: Optional[str] = Form("val_loss"),
    lr_factor: Optional[float] = Form(0.1),
    lr_patience: Optional[int] = Form(3),
    min_lr: Optional[float] = Form(0.00001),
    custom_metrics: Optional[str] = Form(""),
    metrics: Optional[List[str]] = Form(None),
    device: str = Form(...),
):
    print("🔥 /train endpoint tetiklendi")

    # 1️⃣ Dataset'i dosyadan oku
    dataset_path = os.path.join(TEMP_FOLDER, filename)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Veri seti dosyası bulunamadı.")

    df = pd.read_csv(dataset_path)
    print("✅ Veri seti yüklendi:", df.shape)

    # 2️⃣ Layer bilgilerini çöz
    from fastapi import Request
    from starlette.requests import Request  # Eğer yukarıda yoksa

    
    form_data = await request.form()
    layers = []
    i = 0
    while f"layers[{i}][type]" in form_data:
        layer = {
            "type": form_data[f"layers[{i}][type]"],
            "neurons": int(form_data[f"layers[{i}][neurons]"]) if form_data.get(f"layers[{i}][neurons]", "").strip() else None,
            "activation": form_data.get(f"layers[{i}][activation]", ""),
            "kernel_size": form_data.get(f"layers[{i}][kernel_size]", ""),
            "dropout_rate": float(form_data[f"layers[{i}][dropout_rate]"]) if form_data.get(f"layers[{i}][dropout_rate]", "").strip() else None
        }
        layers.append(layer)
        i += 1

    print("🧠 Model katmanları çözüldü:", layers)

    # 3️⃣ JSON konfigürasyonu oluştur
    config = {
        "dataset_path": dataset_path,
        "loss_function": loss_function,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "loss_monitor_freq": loss_monitor_freq,
        "shuffle": shuffle.lower() == "true",
        "device": device,
        "metrics": metrics or [],
        "custom_metrics": [m.strip() for m in custom_metrics.split(",") if m.strip()],
        "layers": layers,
        "callbacks": {
            "early_stopping": {
                "enabled": enable_early_stopping,
                "patience": early_stopping_patience,
                "delta": early_stopping_delta
            },
            "lr_scheduler": {
                "enabled": enable_lr_adapter,
                "monitor": lr_monitor,
                "factor": lr_factor,
                "patience": lr_patience,
                "min_lr": min_lr
            }
        }
    }

    config_path = os.path.join(TEMP_FOLDER, f"{file_id}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        import json
        json.dump(config, f, indent=2)

    print("✅ Konfigürasyon JSON olarak kaydedildi:", config_path)

    return {
        "status": "success",
        "file_id": file_id,
        "config_path": config_path,
        "message": "Model konfigürasyonu oluşturuldu, eğitim için hazır."
    }


# Uygulamayı çalıştırmak için: uvicorn <dosya_adı>:app --reload
