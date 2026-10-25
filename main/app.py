"""
Web UI สำหรับ F1 Singapore GP Winner Prediction
วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ main.py (โฟลเดอร์ main/)
โครงสร้างที่ต้องมี:
    main/
     ├─ main.py
     ├─ app.py          <- ไฟล์นี้
     └─ templates/
         └─ index.html

วิธีรัน (จากโฟลเดอร์ main/):
    pip install flask
    python app.py
แล้วเปิดเบราว์เซอร์ที่ http://127.0.0.1:5000
"""
import sys
import traceback
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ใช้ Logic เดิมจาก main.py ทั้งหมด
import main as pipeline

app = Flask(__name__)


def ensure_data(target_year):
    """ตรวจสอบไฟล์ข้อมูลของปีนั้น ถ้าไม่มีให้ดึงด้วย pulldata.py (ตาม logic ของ main.py)"""
    tel_name = f'singapore_q_{target_year}_data.csv'
    legacy_name = f'singapore_q_top7_{target_year}_data.csv'
    corners_name = f'singapore_{target_year}_corners.csv'

    pipeline.update_config(target_year, None)

    if not pipeline.data_ready(tel_name, legacy_name, corners_name):
        pipeline.run('data/pulldata.py')

    tel_file = pipeline.telemetry_path(tel_name, legacy_name)
    if tel_file is None or not (pipeline.DATA / corners_name).exists():
        raise FileNotFoundError(
            f"ดึงข้อมูลแล้วแต่ยังไม่พบไฟล์ของปี {target_year} ใน {pipeline.DATA}")
    return tel_file, tel_name, legacy_name, corners_name


def parse_year(payload):
    try:
        return int(str(payload.get('year', '')).strip())
    except (TypeError, ValueError):
        raise ValueError("ปีที่ระบุไม่ถูกต้อง กรุณาใส่เป็นตัวเลข ค.ศ. เช่น 2023")


@app.route('/')
def index():
    return render_template('index.html', n_drivers=pipeline.N_DRIVERS)


@app.route('/api/drivers', methods=['POST'])
def api_drivers():
    """รับปี -> ดึง/ตรวจสอบข้อมูล -> คืนรายชื่อนักแข่งทั้งหมด + Top 7 เริ่มต้น"""
    try:
        target_year = parse_year(request.get_json(force=True))
        tel_file, *_ = ensure_data(target_year)
        available = pipeline.drivers_in(tel_file)
        return jsonify({
            'year': target_year,
            'drivers': available,
            'default': available[:pipeline.N_DRIVERS],
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except subprocess.CalledProcessError:
        return jsonify({'error': f'ดึงข้อมูลปีนั้นไม่สำเร็จ (pulldata.py ล้มเหลว) '
                                 f'ตรวจสอบว่าปีนั้นมีการแข่ง Singapore GP และ cache/อินเทอร์เน็ตพร้อมใช้งาน'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """รับปี + รายชื่อนักแข่ง -> รัน pipeline -> คืนผลทำนาย + อันดับจริง + accuracy"""
    try:
        payload = request.get_json(force=True)
        target_year = parse_year(payload)
        raw_drivers = [str(d).strip().upper() for d in payload.get('drivers', []) if str(d).strip()]

        tel_file, tel_name, legacy_name, corners_name = ensure_data(target_year)
        available = pipeline.drivers_in(tel_file)

        # logic เดิมจาก main.py: ถ้าเป็นไฟล์ legacy แล้วนักแข่งที่เลือกไม่อยู่ในไฟล์ -> ดึงใหม่
        if tel_file.name == legacy_name and raw_drivers and not (set(raw_drivers) <= set(available)):
            pipeline.run('data/pulldata.py')
            tel_file = pipeline.telemetry_path(tel_name, legacy_name)
            available = pipeline.drivers_in(tel_file)

        warnings = []
        n = pipeline.N_DRIVERS
        if raw_drivers:
            if len(raw_drivers) != n:
                warnings.append(f"ระบุนักแข่งมา {len(raw_drivers)} คน (ต้องระบุ {n} คน) -> ใช้ Top {n} จากรอบ Qualify แทน")
            elif len(set(raw_drivers)) != n:
                warnings.append(f"มีรายชื่อนักแข่งซ้ำกัน -> ใช้ Top {n} จากรอบ Qualify แทน")
            else:
                unknown = [d for d in raw_drivers if d not in available]
                if unknown:
                    warnings.append(f"ไม่พบนักแข่ง {unknown} ในข้อมูล Qualify -> ใช้ Top {n} จากรอบ Qualify แทน")
        chosen = pipeline.validate_and_select_drivers(raw_drivers, available, n) if raw_drivers else available[:n]

        pipeline.update_config(target_year, chosen)

        import shutil
        shutil.copy(tel_file, pipeline.WORK / tel_name)
        shutil.copy(pipeline.DATA / corners_name, pipeline.WORK / corners_name)

        pipeline.run('matrixP/matrixp.py')
        pipeline.run('weightMatrix/weightMatrix.py')
        pipeline.run('result/result.py')

        pred_file = pipeline.WORK / f'final_race_prediction_{target_year}.csv'
        if not pred_file.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ผลลัพธ์ {pred_file}")

        final_df = pd.read_csv(pred_file)
        final_df = final_df.sort_values(by='Win_Prob_With_SC (%)', ascending=False).reset_index(drop=True)
        predicted_order = final_df['Driver'].tolist()

        actual_ranks = pipeline.get_actual_race_results(target_year, predicted_order)

        rows = []
        for idx, row in final_df.iterrows():
            d = row['Driver']
            act_pos = actual_ranks.get(d, 99)
            rows.append({
                'rank': idx + 1,
                'driver': d,
                'prob_sc': float(row['Win_Prob_With_SC (%)']),
                'prob_nosc': float(row['Win_Prob_No_SC (%)']),
                'actual': int(act_pos) if act_pos != 99 else None,
            })

        accuracy = None
        if actual_ranks:
            acc, concordant, total_pairs = pipeline.calculate_pairwise_accuracy(predicted_order, actual_ranks)
            accuracy = {'percent': round(acc, 2), 'concordant': concordant, 'total_pairs': total_pairs}

        return jsonify({
            'year': target_year,
            'chosen': chosen,
            'warnings': warnings,
            'results': rows,
            'accuracy': accuracy,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'รันสคริปต์ในขั้นตอน pipeline ล้มเหลว: {e}'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # ห้ามเปิด use_reloader: pipeline มีการเขียนไฟล์ config.py ใหม่ตอนรัน
    # ถ้าเปิด reloader ไว้ Flask จะรีสตาร์ทกลางคันทำให้หน้าเว็บขึ้น "Failed to fetch"
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
