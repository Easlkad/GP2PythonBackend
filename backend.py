from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import os
import uuid

app = FastAPI(title="Veri Seti İşleme ve Dönüştürme API")

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
@app.post("/upload")
async def upload_dataset(dataset: UploadFile = File(...)):
    if not dataset.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Sadece CSV dosyaları destekleniyor.")
    
    contents = await dataset.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {str(e)}")
    
    # Sütun isimleri, sütun sayısı ve veri tiplerinin özetini çıkar
    summary = {
        "columns": list(df.columns),
        "column_count": len(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    # Gelecek onay için dosyayı geçici olarak kaydet (benzersiz ID ile)
    file_id = str(uuid.uuid4())
    file_path = save_temp_file(contents, f"{file_id}_{dataset.filename}")
    
    response = {
        "file_id": file_id,
        "filename": dataset.filename,
        "summary": summary,
        "message": "Lütfen yukarıdaki özet bilgileri kontrol edip onaylayın. Onay sonrası veri seti işlenip size gönderilecek."
    }
    return JSONResponse(content=response)

@app.post("/process")
async def process_dataset(file_id: str = Form(...), filename: str = Form(...)):
    file_path = os.path.join(TEMP_FOLDER, f"{file_id}_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {str(e)}")
    
    # Boolean sütunları: "True"/"False" değerlerini 1/0'a dönüştür
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({"True", "False"}):
                df[col] = df[col].map({"True": 1, "False": 0})
    
    # Diğer categorical sütunlar için: 
    # Eğer sütundaki benzersiz değer sayısı 50'nin altında ise, label encoding uygula.
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 50:
            df[col] = pd.Categorical(df[col]).codes
        else:
            # Çok fazla benzersiz değeri olan sütunlar için farklı dönüşüm stratejileri geliştirilebilir.
            pass

    # Eksik değerler için forward fill yöntemi ve tekrarlanan kayıtları kaldırma
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    
    # İşlenmiş veri setini kaydet
    processed_file_path = os.path.join(TEMP_FOLDER, f"processed_{file_id}_{filename}")
    df.to_csv(processed_file_path, index=False)
    
    return FileResponse(processed_file_path, media_type="text/csv", filename=f"processed_{filename}")

# Uygulamayı çalıştırmak için: uvicorn <dosya_adı>:app --reload
