import os
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from Model import Deepnet
import pandas as pd
import tempfile
from typing import List
import io

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

def process_sequences(sequences: List[str], model) -> List[dict]:
    results = []
    for i, seq in enumerate(sequences, 1):
        seq = seq.strip()
        if len(seq) != 41:
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
    return results

@app.post("/predict")
async def predict(
    sequence: str = Form(None),
    file: UploadFile = File(None),
    species: str = Form(...),
    output_format: str = Form("json")  # 新增参数：输出格式
):
    try:
        if species not in species_model_paths:
            return JSONResponse({"error": f"物种 '{species}' 未支持"}, status_code=400)

        model = Deepnet(feature=128, dropout=0.3, filter_num=128, seq_len=41).to(device)
        model.load_state_dict(torch.load(species_model_paths[species], map_location=device))
        model.eval()

        # 僅當 file 存在且有檔名時才進入文件分支
        if file is not None and getattr(file, "filename", None) and file.filename != "":
            content = (await file.read()).decode("utf-8").strip()
            sequences = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith(">")]
            results = process_sequences(sequences, model)

            if output_format == "excel":
                # 创建 DataFrame
                df = pd.DataFrame(results)
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    df.to_excel(tmp.name, index=False)
                # 返回 Excel 文件
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
                return JSONResponse({"error": f"输入序列长度必须为 41 个碱基，当前为 {len(seq)}"}, status_code=400)

            input_tensor = encode_sequence(seq)
            with torch.no_grad():
                output = model(input_tensor).item()
                return {
                    "species": species,
                    "sequence": seq,
                    "probability": round(output, 4),
                    "prediction": "甲基化" if output >= 0.5 else "非甲基化"
                }
        else:
            return JSONResponse({"error": "必须提供 DNA 序列或文件"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 挂载前端静态文件夹
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
