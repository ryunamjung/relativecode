# main.py
from __future__ import annotations

import io
import re
import time
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

EXPECTED_COLS = [
    "선택", "처방코드", "청구코드", "처방명", "항목", "종별가산", "단가", "종별가산단가",
    "1회투", "Tms/Tot Q", "일수", "금액", "급비", "급비지정", "포괄", "완화", "원외", "무료", "처방일자", "항목명"
]

SECTION_ROW_PATTERN = re.compile(r"^\s*\[\s*.+?\s*\]\s*$")  # [ 진찰료 ] 같은 행


# ------------------ utils ------------------
def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _fmt_price(x):
    if pd.isna(x) or x is None:
        return ""
    try:
        fx = float(x)
    except Exception:
        return ""
    if fx.is_integer():
        return f"{int(fx):,}"
    return f"{fx:,.2f}"


def mode_nonempty(s: pd.Series) -> str:
    s = s.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    return s.value_counts().index[0]


def mode_numeric(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    vc = s.value_counts()
    top = vc[vc == vc.max()].index
    return float(sorted(top)[0])


def _clean_lines(raw: str) -> str:
    lines = []
    for ln in raw.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if not ln.strip():
            continue
        if SECTION_ROW_PATTERN.match(ln.strip()):
            continue
        lines.append(ln.lstrip("\t"))
    return "\n".join(lines)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # BOM/공백 제거
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    # 흔한 변형 매핑(필요시 추가)
    rename_map = {
        "처방 코드": "처방코드",
        "청구 코드": "청구코드",
        "처방코드 ": "처방코드",
        "청구코드 ": "청구코드",
        "처 방 코 드": "처방코드",
        "청 구 코 드": "청구코드",
        "처방코드(내부)": "처방코드",
        "청구코드(EDI)": "청구코드",
    }
    return df.rename(columns=rename_map)


def parse_clipboard_tsv(raw: str) -> pd.DataFrame:
    cleaned = _clean_lines(raw)
    if not cleaned.strip():
        return pd.DataFrame(columns=EXPECTED_COLS)

    # 1차: 헤더 있음으로 읽기
    df = pd.read_csv(
        io.StringIO(cleaned),
        sep="\t",
        dtype=str,
        engine="python",
        keep_default_na=False
    )
    df = _normalize_columns(df)

    # 2차: 헤더가 없다고 의심되면(header=None로 재시도)
    if ("처방코드" not in df.columns) and ("청구코드" not in df.columns):
        df2 = pd.read_csv(
            io.StringIO(cleaned),
            sep="\t",
            header=None,
            dtype=str,
            engine="python",
            keep_default_na=False
        )
        df2 = df2.iloc[:, :len(EXPECTED_COLS)]
        df2.columns = EXPECTED_COLS[:df2.shape[1]]
        df = df2
    else:
        # 컬럼 보정: 누락 컬럼 생성
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[EXPECTED_COLS].copy()

    # 숫자 처리
    num_cols = ["종별가산", "단가", "종별가산단가", "1회투", "Tms/Tot Q", "일수", "금액"]
    for c in num_cols:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 날짜 처리
    df["처방일자"] = df["처방일자"].astype(str).str.strip()
    df["처방일자_dt"] = pd.to_datetime(df["처방일자"], format="%Y%m%d", errors="coerce")

    # 섹션 토큰이 처방코드에 들어간 행 제거(예: "[ 진찰료 ]")
    mask_section = df["처방코드"].astype(str).str.strip().str.match(r"^\[.+\]$")
    df = df.loc[~mask_section].copy()

    # 코드 둘 다 비어있는 합계행 제거
    mask_no_codes = (df["처방코드"].astype(str).str.strip() == "") & (df["청구코드"].astype(str).str.strip() == "")
    df = df.loc[~mask_no_codes].copy()

    return df


def get_codes_same_day(
    df_all: pd.DataFrame,
    base_code: str,
    base_col: str,
    code_col: str,
    total_case: int
) -> pd.DataFrame:
    """
    기준코드가 존재하는 (case_id, 처방일자)에서 같은 날 등장한 0401/0801 코드를 집계.
    출력: 항목, 코드(탐색기준), 청구코드, 처방코드, 코드명(처방명), 단가, 급비, rows, case_n, 동반(모든케이스)
    """
    d = df_all.copy()
    if "처방일자" in d.columns:
        d["처방일자"] = d["처방일자"].astype(str).str.strip()
    else:
        d["처방일자"] = ""

    # 방어: 필수 컬럼 없으면 빈 결과
    for col in [base_col, code_col, "case_id", "처방일자", "항목", "처방명", "청구코드", "처방코드"]:
        if col not in d.columns:
            return pd.DataFrame(columns=["항목","코드","청구코드","처방코드","코드명","단가","급비","rows","case_n","동반(모든케이스)"])

    # 기준코드가 있는 (case_id, 처방일자) 키
    hits = d[d[base_col].astype(str).str.strip() == str(base_code).strip()]
    hit_keys = hits[["case_id", "처방일자"]].drop_duplicates()
    if hit_keys.empty:
        return pd.DataFrame(columns=["항목","코드","청구코드","처방코드","코드명","단가","급비","rows","case_n","동반(모든케이스)"])

    merged = d.merge(hit_keys.assign(_hit=1), on=["case_id", "처방일자"], how="inner")

    focus = merged[merged["항목"].isin(["0401", "0801"])].copy()

    # 탐색 기준 코드(청구/처방 선택)
    focus["코드"] = focus[code_col].astype(str).str.strip()

    # 표시용(둘 다 같이)
    focus["청구코드_표시"] = focus["청구코드"].astype(str).str.strip()
    focus["처방코드_표시"] = focus["처방코드"].astype(str).str.strip()

    # 코드명 = 처방명
    focus["코드명"] = focus["처방명"].astype(str).str.strip()

    stats = (
        focus.groupby(["항목", "코드"])
        .agg(
            청구코드=("청구코드_표시", mode_nonempty),
            처방코드=("처방코드_표시", mode_nonempty),
            코드명=("코드명", mode_nonempty),
            단가=("단가", mode_numeric),
            급비=("급비", mode_nonempty),
            rows=("코드", "size"),
            case_n=("case_id", "nunique"),
        )
        .reset_index()
        .sort_values(["항목", "case_n", "rows"], ascending=[True, False, False])
    )

    stats["단가"] = stats["단가"].apply(_fmt_price)

    total_case = int(total_case) if total_case else 0
    stats["동반(모든케이스)"] = stats["case_n"].fillna(0).astype(int).eq(total_case)

    stats = stats[["항목","코드","청구코드","처방코드","코드명","단가","급비","rows","case_n","동반(모든케이스)"]]
    return stats


def build_cases_from_all_rows(df_all: pd.DataFrame) -> pd.DataFrame:
    """all_rows만 있어도 case_id/case_ts로 cases 재구성."""
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["case_id", "case_ts", "rows", "amt"])

    for col in ["case_id", "case_ts"]:
        if col not in df_all.columns:
            return pd.DataFrame(columns=["case_id", "case_ts", "rows", "amt"])

    d = df_all.copy()
    d["case_id"] = d["case_id"].astype(str)
    d["case_ts"] = d["case_ts"].astype(str)

    # amt는 금액 합계로 재구성(없으면 0)
    if "금액" in d.columns:
        amt_series = pd.to_numeric(d["금액"], errors="coerce").fillna(0)
    else:
        amt_series = pd.Series([0] * len(d))

    d["_amt"] = amt_series

    cases = (
        d.groupby(["case_id", "case_ts"], dropna=False)
        .agg(
            rows=("case_id", "size"),
            amt=("_amt", "sum"),
        )
        .reset_index()
        .sort_values("case_ts")
    )

    # 보기 좋게
    cases["rows"] = cases["rows"].astype(int)
    cases["amt"] = cases["amt"].astype(float)

    return cases[["case_id", "case_ts", "rows", "amt"]]


def to_excel_bytes(all_df: pd.DataFrame, cases_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        all_df.to_excel(writer, index=False, sheet_name="all_rows")
        cases_df.to_excel(writer, index=False, sheet_name="cases")
    return output.getvalue()


def load_uploaded_file(file) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    업로드 파일(xlsx/csv)에서 all_rows/cases를 복원.
    - xlsx: sheet 'all_rows', 'cases' 있으면 사용. cases 없으면 all_rows로 재구성.
    - csv: all_rows로 간주, cases는 재구성.
    """
    name = getattr(file, "name", "uploaded")
    lower = name.lower()

    if lower.endswith(".xlsx"):
        data = file.read()
        bio = io.BytesIO(data)
        xls = pd.ExcelFile(bio)

        # all_rows
        if "all_rows" in xls.sheet_names:
            all_df = pd.read_excel(xls, sheet_name="all_rows", dtype=str)
        else:
            # 첫 시트를 all_rows로 간주
            all_df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], dtype=str)

        all_df = _normalize_columns(all_df)

        # 숫자/날짜 컬럼은 다운받았다가 재업로드하면 dtype=str이 될 수 있으니 일부 복구
        for c in ["종별가산","단가","종별가산단가","1회투","Tms/Tot Q","일수","금액"]:
            if c in all_df.columns:
                all_df[c] = all_df[c].astype(str).str.replace(",", "", regex=False).str.strip()
                all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

        if "처방일자" in all_df.columns:
            all_df["처방일자"] = all_df["처방일자"].astype(str).str.strip()
            all_df["처방일자_dt"] = pd.to_datetime(all_df["처방일자"], format="%Y%m%d", errors="coerce")
        else:
            all_df["처방일자_dt"] = pd.NaT

        # 필수 컬럼 보정
        for c in EXPECTED_COLS:
            if c not in all_df.columns:
                all_df[c] = ""
        for c in ["case_id", "case_ts"]:
            if c not in all_df.columns:
                all_df[c] = ""

        # cases
        if "cases" in xls.sheet_names:
            cases_df = pd.read_excel(xls, sheet_name="cases", dtype=str)
            cases_df = _normalize_columns(cases_df)
            # 형태 표준화
            for c in ["case_id", "case_ts", "rows", "amt"]:
                if c not in cases_df.columns:
                    cases_df[c] = ""
            cases_df = cases_df[["case_id", "case_ts", "rows", "amt"]].copy()
            # 숫자 복구
            cases_df["rows"] = pd.to_numeric(cases_df["rows"], errors="coerce").fillna(0).astype(int)
            cases_df["amt"] = pd.to_numeric(cases_df["amt"], errors="coerce").fillna(0.0).astype(float)
        else:
            cases_df = build_cases_from_all_rows(all_df)

        return all_df, cases_df, name

    elif lower.endswith(".csv"):
        data = file.read()
        text = data.decode("utf-8-sig", errors="ignore")
        all_df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        all_df = _normalize_columns(all_df)

        # 숫자/날짜 복구
        for c in ["종별가산","단가","종별가산단가","1회투","Tms/Tot Q","일수","금액"]:
            if c in all_df.columns:
                all_df[c] = all_df[c].astype(str).str.replace(",", "", regex=False).str.strip()
                all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

        if "처방일자" in all_df.columns:
            all_df["처방일자"] = all_df["처방일자"].astype(str).str.strip()
            all_df["처방일자_dt"] = pd.to_datetime(all_df["처방일자"], format="%Y%m%d", errors="coerce")
        else:
            all_df["처방일자_dt"] = pd.NaT

        for c in EXPECTED_COLS:
            if c not in all_df.columns:
                all_df[c] = ""
        for c in ["case_id", "case_ts"]:
            if c not in all_df.columns:
                all_df[c] = ""

        cases_df = build_cases_from_all_rows(all_df)
        return all_df, cases_df, name

    else:
        raise ValueError("xlsx 또는 csv 파일만 업로드할 수 있습니다.")


# ------------------ UI ------------------
st.set_page_config(page_title="0401/0801 규칙 찾기", layout="wide")
st.title("복붙 누적 → 기준코드별 ‘같은 처방일자’ 0401/0801 동반코드(필수 후보 색표시)")

# session_state 초기화
if "all_df" not in st.session_state:
    st.session_state.all_df = pd.DataFrame(columns=EXPECTED_COLS + ["처방일자_dt", "case_id", "case_ts"])
if "cases" not in st.session_state:
    st.session_state.cases = []
if "raw_input" not in st.session_state:
    st.session_state.raw_input = ""
if "last_saved_hash" not in st.session_state:
    st.session_state.last_saved_hash = None

with st.sidebar:
    st.subheader("설정")
    base_col = st.radio("기준코드 컬럼", options=["청구코드", "처방코드"], index=0)
    code_col = st.radio("동반코드 컬럼(나열)", options=["청구코드", "처방코드"], index=0)
    st.caption("추천: 기준코드=청구코드, 동반코드=청구코드 (표시에는 청구/처방/명칭 모두)")

    st.divider()
    st.subheader("누적 불러오기(복원)")
    up = st.file_uploader("이전에 다운로드한 누적 파일 업로드(xlsx/csv)", type=["xlsx", "csv"])
    c_load1, c_load2 = st.columns([1, 1])
    load_mode = c_load1.radio("불러오기 모드", ["덮어쓰기", "추가(append)"], index=0)
    btn_load = c_load2.button("불러오기 실행", use_container_width=True)

    if btn_load:
        if up is None:
            st.warning("업로드 파일을 선택하세요.")
        else:
            try:
                new_all, new_cases_df, fname = load_uploaded_file(up)

                # cases_df -> session list로 변환
                new_cases_list = []
                if not new_cases_df.empty:
                    for _, r in new_cases_df.iterrows():
                        new_cases_list.append({
                            "case_id": str(r.get("case_id", "")),
                            "case_ts": str(r.get("case_ts", "")),
                            "rows": int(r.get("rows", 0) or 0),
                            "amt": float(r.get("amt", 0) or 0),
                        })

                if load_mode == "덮어쓰기":
                    st.session_state.all_df = new_all.copy()
                    st.session_state.cases = new_cases_list
                else:
                    # append: case_id 중복은 그대로 쌓일 수 있으니 주의(그래도 요구사항대로 "다시 읽어서" 유지)
                    st.session_state.all_df = pd.concat([st.session_state.all_df, new_all], ignore_index=True)
                    st.session_state.cases = st.session_state.cases + new_cases_list

                st.session_state.raw_input = ""
                st.session_state.last_saved_hash = None

                st.success(f"불러오기 완료: {fname} / 케이스(저장횟수) {len(st.session_state.cases)}개 / 총 행수 {len(st.session_state.all_df):,}")

            except Exception as e:
                st.error(f"불러오기 실패: {e}")

    st.divider()
    if st.button("전체 초기화(처음부터)", use_container_width=True):
        st.session_state.all_df = pd.DataFrame(columns=EXPECTED_COLS + ["처방일자_dt", "case_id", "case_ts"])
        st.session_state.cases = []
        st.session_state.raw_input = ""
        st.session_state.last_saved_hash = None
        st.success("초기화 완료: 누적/입력 모두 삭제되었습니다.")


tab1, tab2 = st.tabs(["① Paste → 저장(케이스 누적/다운로드)", "② 규칙 탐색(색표시)"])


# ------------------ TAB 1 ------------------
with tab1:
    st.subheader("한 번 복붙 → 저장(=케이스 1개) → 입력창 자동 비움")

    with st.form("paste_form", clear_on_submit=True):
        raw = st.text_area(
            "여기에 표를 그대로 붙여넣기(탭 구분). 저장하면 입력창이 자동으로 비워집니다.",
            height=240,
            key="raw_input",
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        btn_preview = c1.form_submit_button("미리보기(파싱)", use_container_width=True)
        btn_save = c2.form_submit_button("저장", use_container_width=True)
        btn_save_force = c3.form_submit_button("강제 저장(중복허용)", use_container_width=True)

    if btn_preview or btn_save or btn_save_force:
        df_new = parse_clipboard_tsv(raw)
        st.dataframe(df_new.head(30), use_container_width=True)

        total_amt = float(pd.to_numeric(df_new["금액"], errors="coerce").fillna(0).sum()) if not df_new.empty else 0
        st.metric("이번 복붙 총금액", f"{int(total_amt):,}")
        st.metric("이번 복붙 행수", len(df_new))

        if btn_save or btn_save_force:
            if df_new.empty:
                st.warning("저장할 데이터가 없습니다.")
            else:
                h = _hash_text(raw.strip())
                if (not btn_save_force) and (st.session_state.last_saved_hash == h):
                    st.warning("방금 저장한 내용과 동일합니다(중복 저장 방지). 필요하면 '강제 저장'을 누르세요.")
                else:
                    case_id = f"CASE-{time.time_ns()}"
                    case_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    df_new = df_new.copy()
                    df_new["case_id"] = case_id
                    df_new["case_ts"] = case_ts

                    st.session_state.cases.append({
                        "case_id": case_id,
                        "case_ts": case_ts,
                        "rows": int(len(df_new)),
                        "amt": float(total_amt),
                    })
                    st.session_state.all_df = pd.concat([st.session_state.all_df, df_new], ignore_index=True)
                    st.session_state.last_saved_hash = h
                    st.success(f"저장 완료: {case_id} (rows={len(df_new)})")

    # 누적 현황
    all_df = st.session_state.all_df
    st.caption(
        f"현재 누적(=저장횟수): 케이스 {len(st.session_state.cases)}개 / 고유 case_id {all_df['case_id'].nunique() if 'case_id' in all_df.columns else 0}개 / 총 행수 {len(all_df):,}"
    )

    # ✅ 누적 전체 다운로드
    st.divider()
    st.subheader("누적 데이터 다운로드")

    cases_df = pd.DataFrame(st.session_state.cases, columns=["case_id","case_ts","rows","amt"])
    if not all_df.empty:
        dl_all_df = all_df.copy()

        # 정렬(있으면)
        if "case_ts" in dl_all_df.columns and "처방일자_dt" in dl_all_df.columns:
            dl_all_df = dl_all_df.sort_values(["case_ts","처방일자_dt"], ascending=[True, True], na_position="last")

        xlsx_bytes = to_excel_bytes(dl_all_df, cases_df)
        st.download_button(
            label="📥 누적 전체 다운로드 (Excel: all_rows + cases)",
            data=xlsx_bytes,
            file_name=f"누적전체_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        csv_bytes = dl_all_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 누적 전체 다운로드 (CSV: all_rows)",
            data=csv_bytes,
            file_name=f"누적전체_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("아직 누적된 데이터가 없습니다. 먼저 복붙 후 저장하세요.")


# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("규칙 탐색: 기준코드가 있는 ‘같은 처방일자’에 같이 나온 0401/0801 목록")
    st.caption("✅ 연두색 = 기준코드가 등장한 모든 케이스에서 동일하게 동반된 코드(필수 후보)")

    df_all = st.session_state.all_df.copy()

    if df_all.empty:
        st.info("먼저 ① 탭에서 ‘복붙 → 저장’으로 케이스를 누적하거나, 사이드바에서 누적파일을 불러오세요.")
        st.stop()

    # ---------- 안전한 정규화 ----------
    def _norm_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
        s = s.replace({"nan": "", "None": "", "NaN": "", "<NA>": "", "NaT": ""})
        return s

    d = df_all.copy()

    # 중복 컬럼명 방지(혹시라도)
    d = d.loc[:, ~d.columns.duplicated(keep="first")]

    # 필수 컬럼 보장
    for c in ["청구코드", "처방코드", "처방명", base_col]:
        if c not in d.columns:
            d[c] = ""

    # 정규화
    d["청구코드"] = _norm_series(d["청구코드"])
    d["처방코드"] = _norm_series(d["처방코드"])
    d["처방명"] = _norm_series(d["처방명"])
    d[base_col] = _norm_series(d[base_col])

    other_col = "처방코드" if base_col == "청구코드" else "청구코드"

    base_nonempty = int((d[base_col] != "").sum())
    other_nonempty = int((d[other_col] != "").sum())

    # ---------- 후보 키 선택(fallback) ----------
    cand_key_col = base_col
    if base_nonempty == 0 and other_nonempty > 0:
        cand_key_col = other_col
        st.warning(f"⚠️ {base_col} 값이 모두 비어있어서, 임시로 {other_col} 기준으로 후보를 생성합니다.")

    # ---------- cand_src 만들기(컬럼 중복 방지) ----------
    cols = [cand_key_col, "청구코드", "처방코드", "처방명"]
    # 중복 제거(순서 유지)
    seen = set()
    cols = [x for x in cols if not (x in seen or seen.add(x))]

    # ✅ 기준코드 후보는 '항목=0801' 행에서만 뽑기
    # ✅ 기준코드 후보는 '항목=0801' 행에서만 뽑기 (먼저!)
    if "항목" not in d.columns:
        d["항목"] = ""
    d["항목"] = _norm_series(d["항목"])
    d_0801 = d[d["항목"] == "0801"].copy()

    # ✅ 그 다음 cand_src
    cand_src = d_0801.loc[d_0801[cand_key_col] != "", cols].copy()


    if cand_src.empty:
        st.error(
            "기준코드 후보가 0개입니다.\n"
            f"- base_col({base_col}) non-empty: {base_nonempty}\n"
            f"- other_col({other_col}) non-empty: {other_nonempty}\n\n"
            "①탭에서 미리보기로 청구코드/처방코드가 실제 들어오는지 확인하거나,\n"
            "업로드 파일(all_rows)의 컬럼과 값이 정상인지 확인해주세요."
        )
        st.stop()

    # ---------- 후보 요약(cand_df) ----------
    # ⚠️ cand_key_col이 '청구코드'면 agg 결과에 '청구코드'라는 컬럼명을 만들면 reset_index 충돌
    cand_df = (
        cand_src.groupby(cand_key_col, dropna=False)
        .agg(
            rows=(cand_key_col, "size"),
            청구코드_대표=("청구코드", mode_nonempty),
            처방코드_대표=("처방코드", mode_nonempty),
            처방명_대표=("처방명", mode_nonempty),
        )
        .reset_index()
    )

    if cand_df.empty:
        st.error("cand_df 생성 결과가 비었습니다. (데이터를 다시 확인해주세요.)")
        st.stop()

    # ✅ 코드순 정렬
    cand_df[cand_key_col] = _norm_series(cand_df[cand_key_col])
    cand_df = cand_df.sort_values(by=cand_key_col, ascending=False).head(3000)

    # ---------- 드롭다운 라벨 ----------
    label_map = {
        str(r[cand_key_col]): f"{r[cand_key_col]} | 청구:{r['청구코드_대표']} | 처방:{r['처방코드_대표']} | {r['처방명_대표']}"
        for _, r in cand_df.iterrows()
    }

    base_code = st.selectbox(
        "기준코드 선택",
        options=cand_df[cand_key_col].astype(str).tolist(),
        format_func=lambda x: label_map.get(str(x), str(x)),
    )

    # ---------- 실제 규칙 탐색은 base_col 기준 ----------
    # ✅ 실제 규칙 탐색은 '선택한 후보 컬럼(cand_key_col)' 기준으로 일관되게
    hits = df_all[df_all[cand_key_col].astype(str).str.strip() == str(base_code).strip()] if cand_key_col in df_all.columns else df_all.iloc[0:0]
    total_case = int(hits["case_id"].nunique()) if (not hits.empty and "case_id" in hits.columns) else 0
    st.caption(f"기준코드 포함 케이스 수(total_case): {total_case}")

    stats = get_codes_same_day(
        df_all,
        base_code=base_code,
        base_col=cand_key_col,   # ✅ 여기 중요
        code_col=code_col,
        total_case=total_case
    )


    if stats.empty:
        st.warning("이 기준코드가 등장한 케이스/날짜를 아직 찾지 못했습니다.")
        st.stop()

    def highlight_all_cases(row):
        if bool(row.get("동반(모든케이스)")):
            return ["background-color: #d1fae5"] * len(row)
        return [""] * len(row)

    # ✅ 색칠할 컬럼(표시 컬럼)
    view_cols = ["항목","코드","청구코드","처방코드","코드명","단가","급비","rows","case_n"]

    def make_styler(df_part: pd.DataFrame):
        # df_part는 stats에서 항목별로 자른 DF (동반(모든케이스) 포함)
        show = df_part[view_cols].copy()

        # ✅ 칠할 행 마스크
        mask = df_part["동반(모든케이스)"].fillna(False).astype(bool)

        # ✅ 스타일 매트릭스(표시 컬럼만큼) 생성
        style_mat = pd.DataFrame("", index=show.index, columns=show.columns)
        style_mat.loc[mask, :] = "background-color: #d1fae5"

        # ✅ axis=None : 전체 테이블 shape과 동일한 스타일 DF를 반환 (가장 안전)
        sty = show.style.apply(lambda _: style_mat, axis=None)
        return sty

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 0401")
        df_0401 = stats[stats["항목"] == "0401"].copy()
        st.dataframe(make_styler(df_0401), use_container_width=True)

    with colB:
        st.markdown("### 0801")
        df_0801 = stats[stats["항목"] == "0801"].copy()
        st.dataframe(make_styler(df_0801), use_container_width=True)

    st.divider()
    st.subheader("규칙탐색 결과 다운로드")
    out = stats.copy().sort_values(
        ["항목","동반(모든케이스)","case_n","rows"],
        ascending=[True, False, False, False]
    )
    x = io.BytesIO()
    with pd.ExcelWriter(x, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="rules")
    st.download_button(
        "📥 현재 선택 기준코드 규칙 결과 다운로드(Excel)",
        data=x.getvalue(),
        file_name=f"규칙결과_{base_col}_{base_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
