"""Run the whole pipeline interactively:
    python3 main/main.py          # uses the CSV files already in data/
    python3 main/main.py --pull   # download the quali data again first (needs fastf1 + working F1 API)
"""
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'main' else HERE
sys.path.insert(0, str(ROOT))

N_DRIVERS = 7
WORK = ROOT / 'output'
WORK.mkdir(exist_ok=True)
DATA = ROOT / 'data'
CONFIG_PATH = ROOT / 'config.py'


def update_config(year, drivers):
    """อัปเดตไฟล์ config.py เพื่อให้สคริปต์ย่อยอื่นๆ อ่านค่าใหม่ไปใช้งานต่อได้ทันที"""
    content = f"TARGET_YEAR = {year}\nDRIVERS = {repr(drivers)}\n"
    CONFIG_PATH.write_text(content, encoding='utf-8')


def run(script):
    print(f"\n>>> กำลังรัน {script}...")
    subprocess.run([sys.executable, str(ROOT / script)], cwd=WORK, check=True)


def telemetry_path(tel_name, legacy_name):
    for name in (tel_name, legacy_name):
        if (DATA / name).exists():
            return DATA / name
    return None


def drivers_in(path):
    return pd.read_csv(path, usecols=['Driver'])['Driver'].unique().tolist()


def data_ready(tel_name, legacy_name, corners_name):
    """ตรวจสอบว่ามีไฟล์ข้อมูลของปีนั้นพร้อมอยู่แล้วหรือไม่"""
    if not (DATA / corners_name).exists():
        return False
    if (DATA / tel_name).exists() or (DATA / legacy_name).exists():
        return True
    return False


def validate_and_select_drivers(raw_drivers, available, n_drivers=N_DRIVERS):
    """
    ตรวจสอบรายชื่อนักแข่งที่รับเข้ามา:
    - หากใส่มาไม่ครบ 7 คน, เกิน 7 คน, มีชื่อซ้ำ หรือไม่มีชื่อในข้อมูลปีนั้น
    - จะ Fallback ไปใช้นักแข่ง Top 7 จากรอบ Qualify (available[:7]) แทนทันที
    """
    if len(raw_drivers) != n_drivers:
        print(f"⚠️ ระบุนักแข่งมา {len(raw_drivers)} คน (ต้องระบุ {n_drivers} คน) -> ใช้ Top {n_drivers} จากรอบ Qualify แทน")
        return available[:n_drivers]

    if len(set(raw_drivers)) != n_drivers:
        print(f"⚠️ มีรายชื่อนักแข่งซ้ำกัน -> ใช้ Top {n_drivers} จากรอบ Qualify แทน")
        return available[:n_drivers]

    unknown = [d for d in raw_drivers if d not in available]
    if unknown:
        print(f"⚠️ ไม่พบนักแข่ง {unknown} ในข้อมูล Qualify -> ใช้ Top {n_drivers} จากรอบ Qualify แทน")
        return available[:n_drivers]

    return list(raw_drivers)


def get_actual_race_results(year, drivers):
    """ดึงผลการแข่งขันจริง (Race) ของสนาม Singapore จาก FastF1"""
    try:
        import fastf1
        cache_dir = WORK / 'f1_cache'
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

        session = fastf1.get_session(year, 'Singapore', 'R')
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results

        actual_ranks = {}
        for d in drivers:
            row = results[results['Abbreviation'] == d]
            if not row.empty:
                pos = row['Position'].values[0]
                actual_ranks[d] = int(pos) if pd.notna(pos) else 99
            else:
                actual_ranks[d] = 99
        return actual_ranks
    except Exception as e:
        print(f"⚠️ ไม่สามารถดึงผลการแข่งขันจริงจาก FastF1 ได้: {e}")
        return {}


def calculate_pairwise_accuracy(predicted_order, actual_ranks):
    """
    คำนวณความถูกต้องด้วยวิธี Pairwise Concordance (Kendall's Tau Rank Accuracy):
    เปรียบเทียบนักแข่งทุกคู่ (i, j) ที่โมเดลทำนายว่า i มีโอกาสชนะมากกว่า j (i มาก่อน j)
    แล้วตรวจดูว่าในสนามจริง i จบอันดับดีกว่า (ตัวเลขอันดับน้อยกว่า) j จริงหรือไม่
    """
    concordant = 0
    total_pairs = 0
    n = len(predicted_order)

    for i in range(n):
        for j in range(i + 1, n):
            d_higher = predicted_order[i]  # นักแข่งที่โมเดลให้เปอร์เซ็นต์สูงกว่า
            d_lower = predicted_order[j]   # นักแข่งที่โมเดลให้เปอร์เซ็นต์ต่ำกว่า

            if d_higher in actual_ranks and d_lower in actual_ranks:
                total_pairs += 1
                if actual_ranks[d_higher] < actual_ranks[d_lower]:
                    concordant += 1
                elif actual_ranks[d_higher] == actual_ranks[d_lower]:
                    concordant += 0.5  # กรณีอันดับเท่ากัน

    accuracy = (concordant / total_pairs * 100) if total_pairs > 0 else 0.0
    return accuracy, concordant, total_pairs


def main():
    # ================= 1. รับค่าปี (Year) จากผู้ใช้ =================
    year_input = input("กรุณาใส่ปีที่ต้องการทราบผล (เช่น 2023): ").strip()
    try:
        target_year = int(year_input)
    except ValueError:
        sys.exit("❌ ปีที่ระบุไม่ถูกต้อง กรุณาใส่เป็นตัวเลข ค.ศ.")

    # อัปเดต config เบื้องต้นให้ pulldata.py รู้ว่าต้องดึงข้อมูลของปีไหน
    update_config(target_year, None)

    tel_name = f'singapore_q_{target_year}_data.csv'
    legacy_name = f'singapore_q_top7_{target_year}_data.csv'
    corners_name = f'singapore_{target_year}_corners.csv'

    # ================= 2. ตรวจสอบและดึงข้อมูลของปีนั้นก่อน =================
    if '--pull' in sys.argv:
        run('data/pulldata.py')
    elif not data_ready(tel_name, legacy_name, corners_name):
        print(f"ไฟล์ข้อมูลของปี {target_year} ยังไม่มี -> ดึงข้อมูลอัตโนมัติ")
        run('data/pulldata.py')

    tel_file = telemetry_path(tel_name, legacy_name)
    if tel_file is None or not (DATA / corners_name).exists():
        sys.exit(f"\n❌ ดึงข้อมูลแล้วแต่ยังไม่พบไฟล์ของปี {target_year} ใน {DATA}")

    # ================= 3. แสดงรายชื่อนักแข่งทั้งหมดในปีนั้น แล้วค่อยถาม Input =================
    available = drivers_in(tel_file)
    print("\n" + "-" * 65)
    print(f"📋 รายชื่อนักแข่งทั้งหมดในปี {target_year} (เรียงตามอันดับ Qualify):")
    print("   " + ", ".join(available))
    print(f"⭐ Top {N_DRIVERS} Qualify (ค่าเริ่มต้นหากไม่ระบุ/ระบุไม่ครบ): {', '.join(available[:N_DRIVERS])}")
    print("-" * 65)

    drivers_input = input(
        f"กรุณาใส่ตัวย่อนักแข่ง {N_DRIVERS} คน (คั่นด้วยเว้นวรรค หรือกด Enter เพื่อใช้ Top {N_DRIVERS}): "
    ).strip()
    raw_drivers = [d.strip().upper() for d in drivers_input.replace(',', ' ').split() if d.strip()]

    # หากพบว่าใช้ไฟล์แบบเก่า (legacy) ที่มีนักแข่งไม่ครบตามที่ผู้ใช้เลือก ให้สั่งดึงข้อมูลใหม่
    if tel_file.name == legacy_name and raw_drivers and not (set(raw_drivers) <= set(available)):
        print(f"นักแข่งที่เลือกไม่มีในไฟล์ {legacy_name} -> กำลังดึงข้อมูลนักแข่งทั้งหมดใหม่...")
        run('data/pulldata.py')
        tel_file = telemetry_path(tel_name, legacy_name)
        available = drivers_in(tel_file)

    chosen = validate_and_select_drivers(raw_drivers, available, N_DRIVERS)

    # อัปเดต config.py ด้วยรายชื่อนักแข่ง 7 คนที่ผ่านการตรวจสอบแล้ว
    update_config(target_year, chosen)
    print(f"\n✅ นักแข่งที่นำมาเปรียบเทียบ ({N_DRIVERS} คน): {chosen}")

    shutil.copy(tel_file, WORK / tel_name)
    shutil.copy(DATA / corners_name, WORK / corners_name)

    # ================= 4. รัน Pipeline คำนวณ =================
    run('matrixP/matrixp.py')
    run('weightMatrix/weightMatrix.py')
    run('result/result.py')

    # ================= 5. อ่านผลจาก result.py + เทียบอันดับจริง =================
    pred_file = WORK / f'final_race_prediction_{target_year}.csv'
    if not pred_file.exists():
        sys.exit(f"❌ ไม่พบไฟล์ผลลัพธ์ {pred_file}")

    final_df = pd.read_csv(pred_file)
    final_df = final_df.sort_values(by='Win_Prob_With_SC (%)', ascending=False).reset_index(drop=True)
    predicted_order = final_df['Driver'].tolist()

    actual_ranks = get_actual_race_results(target_year, predicted_order)

    print("\n" + "=" * 78)
    print(f"🏁 สรุปผลการทำนายและอันดับจริง Singapore GP ({target_year})")
    print("=" * 78)
    print(f"{'อันดับทำนาย':<12} | {'นักแข่ง':<8} | {'โอกาสชนะ (SC %)':<17} | {'โอกาสชนะ (No SC %)':<19} | {'อันดับจริง (Race)':<15}")
    print("-" * 78)

    for idx, row in final_df.iterrows():
        d = row['Driver']
        prob_sc = row['Win_Prob_With_SC (%)']
        prob_nosc = row['Win_Prob_No_SC (%)']
        act_pos = actual_ranks.get(d, 99)
        act_str = f"อันดับ {act_pos}" if act_pos != 99 else "DNF / ไม่พบข้อมูล"
        print(f"{idx + 1:<12} | {d:<8} | {prob_sc:>14.2f}%   | {prob_nosc:>16.2f}%   | {act_str:<15}")

    if actual_ranks:
        acc, concordant, total_pairs = calculate_pairwise_accuracy(predicted_order, actual_ranks)
        print("=" * 78)
        print(f"🎯 ความถูกต้องของการเรียงลำดับ (Pairwise Accuracy): {acc:.2f}%")
        print(f"   (ทายลำดับเปรียบเทียบถูกต้อง {concordant:g} คู่ จากทั้งหมด {total_pairs} คู่)")
    print("=" * 78)
    print(f"\nไฟล์ผลลัพธ์ทั้งหมดถูกบันทึกไว้ที่: {WORK}")


if __name__ == '__main__':
    main()