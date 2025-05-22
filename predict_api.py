import os
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from Model import Deepnet
import pandas as pd
import tempfile
from typing import List, Dict
import io
import logging
from functools import lru_cache
import time

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 強制使用 CPU
device = torch.device("cpu")

# 模型緩存
model_cache: Dict[str, Deepnet] = {}

# 自动扫描模型目录，构建物种-模型文件路径映射
def load_available_models(model_dir="DNA_model"):
    species_map = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".pt"):
            species_name = filename.replace("DNA_model_", "").replace(".pt", "")
            species_map[species_name] = os.path.join(model_dir, filename)
    return species_map

species_model_paths = load_available_models()

@app.get("/species")
def get_species_list():
    return list(species_model_paths.keys())

# 编码序列函数，碱基转换成数字索引
def encode_sequence(seq):
    table = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    try:
        return torch.tensor([table[c.upper()] for c in seq], dtype=torch.long, device=device).unsqueeze(0)
    except KeyError as e:
        raise ValueError(f"无效碱基字符：{e}")

def get_model(species: str) -> Deepnet:
    """獲取模型，如果緩存中沒有則載入"""
    if species not in model_cache:
        try:
            model = Deepnet(feature=128, dropout=0.3, filter_num=128, seq_len=41).to(device)
            model.load_state_dict(torch.load(species_model_paths[species], map_location=device))
            model.eval()
            model_cache[species] = model
            logger.info(f"Loaded model for species: {species}")
        except Exception as e:
            logger.error(f"Error loading model for species {species}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"模型載入失敗：{str(e)}")
    return model_cache[species]

def process_sequences(sequences: List[str], model) -> List[dict]:
    results = []
    for i, seq in enumerate(sequences, 1):
        try:
            seq = seq.strip()
            if len(seq) != 41:
                logger.warning(f"Invalid sequence length {len(seq)} at position {i}")
                continue
            
            input_tensor = encode_sequence(seq)
            with torch.no_grad():
                output = model(input_tensor).item()
                results.append({
                    "sequence_number": i,
                    "sequence": seq,
                    "probability": round(output, 4),
                    "prediction": "甲基化" if output >= 0.5 else "非甲基化"
                })
        except Exception as e:
            logger.error(f"Error processing sequence at position {i}: {str(e)}")
            continue
    return results

@app.post("/predict")
async def predict(
    sequence: str = Form(None),
    file: UploadFile = File(None),
    species: str = Form(...),
    output_format: str = Form("json")
):
    start_time = time.time()
    try:
        if species not in species_model_paths:
            raise HTTPException(status_code=400, detail=f"物種 '{species}' 未支持")

        # 獲取模型（使用緩存）
        model = get_model(species)

        # 僅當 file 存在且有檔名時才進入文件分支
        if file is not None and getattr(file, "filename", None) and file.filename != "":
            content = (await file.read()).decode("utf-8").strip()
            sequences = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith(">")]
            results = process_sequences(sequences, model)

            if output_format == "excel":
                df = pd.DataFrame(results)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    df.to_excel(tmp.name, index=False)
                return FileResponse(
                    tmp.name,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    filename=f"{species}_methylation_predictions.xlsx"
                )
            else:
                return results

        # 單序列分支
        elif sequence:
            seq = sequence.strip()
            if len(seq) != 41:
                raise HTTPException(status_code=400, detail=f"輸入序列長度必須為 41 個鹼基，當前為 {len(seq)}")

            input_tensor = encode_sequence(seq)
            with torch.no_grad():
                output = model(input_tensor).item()
                result = {
                    "species": species,
                    "sequence": seq,
                    "probability": round(output, 4),
                    "prediction": "甲基化" if output >= 0.5 else "非甲基化"
                }
                logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
                return result
        else:
            raise HTTPException(status_code=400, detail="必須提供 DNA 序列或文件")
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 挂载前端静态文件夹
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
