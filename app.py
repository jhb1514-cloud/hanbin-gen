import streamlit as st
import os
import sys
import asyncio
import re
import random
import requests
import numpy as np
import nest_asyncio
import edge_tts

nest_asyncio.apply()
from google import genai
from groq import Groq
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, VideoClip

# ── 시스템 설정 ──
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

st.set_page_config(page_title="한빈 AI 유튜브 스튜디오", layout="wide")

# ── API 키 로드 (로컬: api.env / 배포: Streamlit Secrets) ──
def _load_keys():
    # Streamlit Cloud Secrets 우선
    try:
        return st.secrets["gemini"], st.secrets.get("groq", "")
    except Exception:
        pass
    # 로컬 api.env 폴백
    keys = {}
    try:
        with open("api.env") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    keys[k.lower()] = v
    except FileNotFoundError:
        pass
    return keys.get("gemini", ""), keys.get("groq", "")

GEMINI_KEY, GROQ_KEY = _load_keys()

gemini_client = genai.Client(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

if not os.path.exists("output"):
    os.makedirs("output")

# ════════════════════════════════════════════
# 🔐 로그인 화면
# ════════════════════════════════════════════
ACCESS_CODE = "626800"

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.markdown("""
        <div style='text-align:center; padding: 80px 0 20px 0'>
            <h1>🎬 한빈 AI 유튜브 스튜디오</h1>
            <p style='color:gray; font-size:16px'>액세스 코드를 입력하세요</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            code_input = st.text_input("", type="password", placeholder="코드 입력", label_visibility="collapsed")
            if st.form_submit_button("입장하기", use_container_width=True, type="primary"):
                if code_input == ACCESS_CODE:
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("코드가 올바르지 않습니다.")
    st.stop()

# ════════════════════════════════════════════
# 유틸 함수
# ════════════════════════════════════════════

def ask_ai(prompt, system_instruction="", model="gemini-2.5-flash"):
    """범용 AI 호출"""
    try:
        full = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        res = gemini_client.models.generate_content(model=model, contents=full)
        return res.text, model.split("/")[-1]
    except Exception as gemini_err:
        if groq_client is not None:
            try:
                res = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": system_instruction},
                              {"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                )
                return res.choices[0].message.content, "Groq"
            except Exception as groq_err:
                return f"오류: {str(groq_err)}", "에러"
        return f"오류: {str(gemini_err)}", "에러"


def ask_visual_ai(prompt):
    """나노바나나 모델 - 비주얼/영상 프롬프트 전용, 실패 시 gemini-2.5-flash 폴백"""
    res, engine = ask_ai(prompt, model="nano-banana-pro-preview")
    if engine == "에러":
        res, engine = ask_ai(prompt, model="gemini-2.5-flash")
    return res, engine


def clean_script_for_tts(script):
    t = script
    # ── 무대 지시사항 제거 (읽으면 안 되는 부분) ──
    # "**[23-31초] 3위: 푸들 (화면: ...)**" 같은 줄 전체 제거
    t = re.sub(r'^\s*\*+\s*\[.*?\].*$', '', t, flags=re.MULTILINE)
    # "[23-31초]", "[30초~]" 등 시간 마커 + 그 줄의 나머지
    t = re.sub(r'\[\d+[\-~]\d*초[^\]]*\][^\n]*', '', t)
    t = re.sub(r'\[\d+초[^\]]*\][^\n]*', '', t)
    # "나레이션 (밝고 활기찬 톤):" 등 나레이션 지시사항
    t = re.sub(r'\**나레이션\s*\([^)]*\)\s*:?\s*\**', '', t)
    t = re.sub(r'\*+나레이션\s*:\s*\**', '', t)
    # "(화면: 다양한 크기의 푸들...)" 화면 설명
    t = re.sub(r'\(화면\s*:[^)]*\)', '', t)
    t = re.sub(r'\(화면[^)]*\)', '', t)
    # "(내레이션)", "(자막)", "(효과음)" 등 괄호 지시어
    t = re.sub(r'\((내레이션|자막|효과음|음악|BGM|나레이션)[^)]*\)', '', t)
    # ── 마크다운 서식 제거 ──
    t = re.sub(r'\*+', '', t)
    t = re.sub(r'#+\s*', '', t)
    t = re.sub(r'_{1,3}', '', t)
    t = re.sub(r'`+', '', t)
    t = re.sub(r'~+', '', t)
    t = re.sub(r'[•·▶▷→✅❌🔥💡]\s*', '', t)
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'[「」『』【】《》\[\]<>]', '', t)
    t = re.sub(r'[!]{2,}', '!', t)
    t = re.sub(r'[?]{2,}', '?', t)
    # 이모지만 제거
    t = re.sub(r'[\U00010000-\U0010ffff]', '', t, flags=re.UNICODE)
    t = re.sub(r'\s+', ' ', t)
    result = t.strip()
    if len(result) < 10:
        result = re.sub(r'\s+', ' ', script).strip()
    return result


def create_voice_sync(text, voice_name, rate="-5%", pitch="+0Hz"):
    """별도 Python 프로세스로 edge-tts 실행 (Streamlit 이벤트 루프 완전 분리)"""
    import subprocess, json

    clean = clean_script_for_tts(text)
    if not clean:
        raise ValueError("정리된 텍스트가 비어 있습니다.")

    # rate 정규화
    if rate and not rate.startswith(('+', '-')):
        rate = '+' + rate

    # 안전 문자만 남기기
    safe = re.sub(r'[^\uAC00-\uD7A3\u3131-\u318E\u3200-\u321E\u3260-\u327E'
                  r'a-zA-Z0-9\s.,!?:;\'\"()\-]', ' ', clean)
    safe = re.sub(r'\s+', ' ', safe).strip()

    if len(safe) < 3:
        raise ValueError(f"정리 후 텍스트가 너무 짧습니다.")

    payload = json.dumps({
        'text': safe, 'voice': voice_name,
        'rate': rate, 'pitch': pitch,
        'out': 'output/voice.mp3'
    }, ensure_ascii=False)

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts_worker.py')
    result = subprocess.run(
        [sys.executable, worker],
        input=payload.encode('utf-8'),
        capture_output=True,
        timeout=90,
        env={**os.environ, 'PYTHONUTF8': '1'}
    )

    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(err or "음성 생성 실패")
    if not os.path.exists("output/voice.mp3"):
        raise RuntimeError("음성 파일이 생성되지 않았습니다.")


def split_into_caption_groups(text, max_chars=18):
    """대본을 자막 단위로 분리: 문장 경계 우선, 긴 문장은 쉼표·어절 기준으로 추가 분리"""
    # 1단계: 문장 분리 (마침표·느낌표·물음표·줄바꿈)
    raw = re.split(r'(?<=[.!?。！？])\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]

    groups = []
    for sent in sentences:
        if len(sent) <= max_chars:
            groups.append(sent)
        else:
            # 2단계: 쉼표·접속사로 분리
            parts = re.split(r'[,，、]\s*', sent)
            current = ""
            for part in parts:
                candidate = (current + " " + part).strip() if current else part
                if current and len(candidate) > max_chars:
                    groups.append(current.strip())
                    current = part
                else:
                    current = candidate
            if current.strip():
                groups.append(current.strip())

    # 3단계: 그래도 긴 그룹은 단어(어절) 기준으로 강제 분리
    final = []
    for g in groups:
        if len(g) <= max_chars:
            final.append(g)
        else:
            words = g.split()
            chunk = []
            for w in words:
                chunk.append(w)
                if len(' '.join(chunk)) >= max_chars:
                    final.append(' '.join(chunk))
                    chunk = []
            if chunk:
                final.append(' '.join(chunk))

    return [g for g in final if g.strip()]


def generate_srt(script, chars_per_second=4.5):
    sentences = re.split(r'(?<=[.!?。\n])\s*', script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    lines, t = [], 0.0
    for i, s in enumerate(sentences, 1):
        dur = max(len(s) / chars_per_second, 1.5)
        def fmt(sec):
            h,m,ss=int(sec//3600),int((sec%3600)//60),int(sec%60)
            ms=int((sec-int(sec))*1000)
            return f"{h:02}:{m:02}:{ss:02},{ms:03}"
        lines += [str(i), f"{fmt(t)} --> {fmt(t+dur)}", s, ""]
        t += dur + 0.1
    return "\n".join(lines)


_POLL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_ai_bg(prompt_en, width=1280, height=720, seed=42):
    """AI 배경 이미지: Gemini Flash 이미지생성 → Picsum 실사 사진
    반환: (PIL.Image or None, 소스/오류 str)
    """
    errors = {}

    # ── 1차: Gemini Flash 이미지 생성 (표준 API 지원) ──
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=f"Create a cinematic 16:9 YouTube background image: {prompt_en}",
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                if img.size != (width, height):
                    img = img.resize((width, height), Image.LANCZOS)
                return img, "Gemini Flash"
        errors["gemini"] = "이미지 파트 없음"
    except Exception as e:
        errors["gemini"] = f"{type(e).__name__}: {str(e)[:80]}"

    # ── 2차: Picsum 실사 사진 (확인된 안정 소스) ──
    try:
        url = f"https://picsum.photos/seed/{seed}/{width}/{height}"
        r = requests.get(url, headers=_POLL_HEADERS, timeout=20, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5000:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            return img, "Picsum"
        errors["picsum"] = f"HTTP {r.status_code}"
    except Exception as e:
        errors["picsum"] = f"{type(e).__name__}: {str(e)[:80]}"

    return None, " | ".join(f"{k}:{v}" for k, v in errors.items())


def make_gradient_bg(width, height, seed=0):
    palettes = [
        [(30,20,80),(110,40,180)],  # 보라
        [(15,80,40),(30,160,90)],   # 초록
        [(120,25,40),(200,80,20)],  # 붉은 오렌지
        [(20,70,150),(30,140,210)], # 파랑
        [(110,70,10),(200,140,25)], # 골드
        [(70,15,110),(150,30,160)], # 마젠타
    ]
    c1, c2 = palettes[seed % len(palettes)]
    r = np.linspace(c1[0],c2[0],height).reshape(-1,1)*np.ones((1,width))
    g = np.linspace(c1[1],c2[1],height).reshape(-1,1)*np.ones((1,width))
    b = np.linspace(c1[2],c2[2],height).reshape(-1,1)*np.ones((1,width))
    return Image.fromarray(np.stack([r,g,b],axis=2).astype(np.uint8))


def add_vignette(img):
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for i in range(min(W,H)//2):
        alpha = int(255*(i/(min(W,H)//2))**1.5)
        draw.ellipse([i,i,W-i,H-i], fill=alpha)
    vignette = Image.new("RGB", (W,H), (0,0,0))
    return Image.composite(img, vignette, mask)


def make_ken_burns_frame(img_pil, t, duration, W=1280, H=720, zoom_to=1.08):
    progress = t / max(duration, 0.001)
    zoom = 1.0 + (zoom_to - 1.0) * progress
    new_w, new_h = int(W*zoom), int(H*zoom)
    resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w-W)//2, (new_h-H)//2
    return resized.crop((left, top, left+W, top+H))


def draw_youtube_caption(bg_img, text, font_bold, W=1280, H=720):
    """YouTube Shorts 스타일 자막 — 자동 2줄 줄바꿈, 화면 밖 넘침 방지"""
    img = bg_img.copy()
    draw = ImageDraw.Draw(img)
    words = text.split()
    if not words:
        return img

    YELLOW, WHITE, BLACK = (255,220,0), (255,255,255), (0,0,0)
    OUTLINE = 4
    MAX_W = int(W * 0.86)

    def measure(s):
        bb = draw.textbbox((0, 0), s, font=font_bold)
        return bb[2] - bb[0], bb[3] - bb[1]

    # 1줄로 맞는지 확인
    full_w, full_h = measure(text)
    if full_w <= MAX_W:
        lines = [text]
    else:
        # 절반 위치에서 분할
        mid = len(words) // 2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]

    line_h = full_h + 6
    total_h = line_h * len(lines)
    pad = 20
    base_y = int(H * 0.63) - total_h // 2

    # 배경 박스 — 가장 긴 줄 기준
    max_lw = max(measure(l)[0] for l in lines)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)
    bx0 = W // 2 - max_lw // 2 - pad
    bx1 = W // 2 + max_lw // 2 + pad
    by0 = base_y - pad // 2
    by1 = base_y + total_h + pad // 2
    ov.rounded_rectangle([bx0, by0, bx1, by1], radius=14, fill=(0, 0, 0, 155))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 각 줄 그리기
    for li, line in enumerate(lines):
        lw, _ = measure(line)
        lx = (W - lw) // 2
        ly = base_y + li * line_h
        lwords = line.split()
        cx = lx
        for wi, word in enumerate(lwords):
            color = YELLOW if (li == 0 and wi == 0) else WHITE
            ws = word + (" " if wi < len(lwords) - 1 else "")
            ww, _ = measure(ws)
            for dx in range(-OUTLINE, OUTLINE + 1):
                for dy in range(-OUTLINE, OUTLINE + 1):
                    if abs(dx) + abs(dy) >= OUTLINE:
                        draw.text((cx + dx, ly + dy), ws, font=font_bold, fill=BLACK)
            draw.text((cx, ly), ws, font=font_bold, fill=color)
            cx += ww
    return img


def create_auto_video(script, topic="", use_ai_bg=True, bg_prompt="",
                      num_ai_bgs=4, progress_cb=None):
    W, H = 1280, 720

    clean = clean_script_for_tts(script)
    if not clean.strip():
        raise ValueError("대본이 비어 있습니다.")

    # 문장/구절 단위로 자막 그룹 분리 (어절 단위 강제 분할보다 훨씬 자연스러움)
    groups = split_into_caption_groups(clean, max_chars=18)
    if not groups:
        raise ValueError("대본이 비어 있습니다.")

    if os.path.exists("output/voice.mp3"):
        audio = AudioFileClip("output/voice.mp3")
        audio_dur = audio.duration
        audio.close()
    else:
        audio_dur = sum(len(g) for g in groups) * 0.12  # 글자수 기반 추정

    # Fix 4: 글자 수 비례로 자막 싱크 맞춤
    total_chars = sum(len(g) for g in groups)
    if total_chars > 0:
        durations = [max(audio_dur * len(g) / total_chars, 0.2) for g in groups]
    else:
        durations = [audio_dur / len(groups)] * len(groups)
    dur_per_group = audio_dur / len(groups)  # 씬 배분용 평균값 유지

    # 나노바나나로 배경 프롬프트 생성 (더 창의적인 결과)
    if progress_cb:
        progress_cb(0.01, "🤖 나노바나나로 씬 프롬프트 생성 중...")

    scene_prompts = []
    if use_ai_bg and not bg_prompt:
        try:
            nb_prompt = (
                f"유튜브 영상 주제: {topic or script[:100]}\n\n"
                f"이 주제에 맞는 서로 다른 cinematic 배경 이미지 영문 프롬프트 {num_ai_bgs}개를 "
                f"한 줄씩 번호 없이 써줘. 각 프롬프트는 15단어 이내, 영어만."
            )
            nb_res, _ = ask_visual_ai(nb_prompt)
            scene_prompts = [l.strip() for l in nb_res.strip().split('\n') if l.strip()][:num_ai_bgs]
        except:
            pass

    if not scene_prompts:
        base = bg_prompt if bg_prompt else f"cinematic 4K youtube video background, dramatic lighting, {topic}"
        scene_prompts = [f"{base}, scene {i+1}" for i in range(num_ai_bgs)]

    # 배경 이미지 생성
    backgrounds = []
    for s, prompt in enumerate(scene_prompts):
        if progress_cb:
            progress_cb(0.02 + 0.28 * s / num_ai_bgs, f"🖼️ 배경 {s+1}/{num_ai_bgs} 생성 중...")
        if use_ai_bg:
            bg, src = fetch_ai_bg(prompt, W, H, seed=s*137+42)
        else:
            bg, src = None, "없음"
        if bg is None:
            if progress_cb:
                progress_cb(
                    0.02 + 0.28 * (s + 1) / num_ai_bgs,
                    f"⚠️ 배경 {s+1} AI 실패 → 그라데이션 사용\n오류: {src}"
                )
            bg = make_gradient_bg(W, H, seed=s)
            src = "그라데이션"
        else:
            if progress_cb:
                progress_cb(
                    0.02 + 0.28 * (s + 1) / num_ai_bgs,
                    f"✅ 배경 {s+1}/{num_ai_bgs} [{src}] 완료"
                )
        bg = add_vignette(bg)
        # Fix 5: 텍스트 가독성용 가벼운 어둡힘 (너무 어둡지 않게 0.2)
        dim = Image.new("RGB", (W, H), (0, 0, 0))
        bg = Image.blend(bg, dim, alpha=0.2)
        backgrounds.append(bg)

    # 폰트
    try:
        font_bold = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 76)
    except:
        try:
            font_bold = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 76)
        except:
            font_bold = ImageFont.load_default()

    groups_per_scene = max(1, len(groups) // num_ai_bgs)

    clips = []
    for i, group in enumerate(groups):
        if progress_cb:
            progress_cb(0.30 + 0.65 * i / len(groups),
                        f"🎬 프레임 {i+1}/{len(groups)} 렌더링 중...")
        bg_idx = min(i // groups_per_scene, len(backgrounds)-1)
        bg_pil = backgrounds[bg_idx]
        t_offset = (i % groups_per_scene) * dur_per_group
        scene_dur = groups_per_scene * dur_per_group

        # Fix 3: 정적 프레임 1장 미리 렌더링 → ImageClip (VideoClip 콜백 방식보다 수십 배 빠름)
        mid_t = t_offset + durations[i] / 2
        zoomed = make_ken_burns_frame(bg_pil, mid_t, scene_dur, W, H, zoom_to=1.08)
        framed = draw_youtube_caption(zoomed, group, font_bold, W, H)
        clips.append(ImageClip(np.array(framed), duration=durations[i]))

    if progress_cb:
        progress_cb(0.95, "🎵 음성 합치는 중...")

    video = concatenate_videoclips(clips, method="compose")

    if os.path.exists("output/voice.mp3"):
        audio = AudioFileClip("output/voice.mp3")
        final_dur = min(audio.duration, video.duration)
        video = video.subclipped(0, final_dur).with_audio(audio.subclipped(0, final_dur))

    out_path = "output/auto_video.mp4"
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    video.close()

    if progress_cb:
        progress_cb(1.0, "✅ 완료!")
    return out_path


def create_desktop_shortcut():
    """현재 사용자 바탕화면에 실행 바로가기 생성 (VBScript 방식, 추가 패키지 불필요)"""
    import subprocess, tempfile
    app_dir = os.path.dirname(os.path.abspath(__file__))
    bat_path = os.path.join(app_dir, "start_studio.bat").replace("/", "\\")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    lnk_path = os.path.join(desktop, "한빈 AI 유튜브 스튜디오.lnk").replace("/", "\\")

    vbs = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
Set oLink = oWS.CreateShortcut("{lnk_path}")
oLink.TargetPath = "{bat_path}"
oLink.WorkingDirectory = "{app_dir.replace('/', chr(92))}"
oLink.WindowStyle = 1
oLink.Description = "한빈 AI 유튜브 스튜디오"
oLink.Save
"""
    tmp = tempfile.NamedTemporaryFile(suffix=".vbs", delete=False, mode="w", encoding="utf-8")
    tmp.write(vbs)
    tmp.close()
    try:
        subprocess.run(["cscript", "//nologo", tmp.name], capture_output=True, timeout=10)
    finally:
        os.remove(tmp.name)
    return os.path.exists(lnk_path)


# ════════════════════════════════════════════
# 세션 상태
# ════════════════════════════════════════════
for k, v in [('script',''),('bgm_recommendations',''),('topic',''),('bg_prompt_auto',''),
             ('thumb_text',''),('thumb_img_prompt','')]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════
st.sidebar.title("🎬 유튜브 제작 센터")
if st.sidebar.button("🚪 로그아웃", use_container_width=True):
    st.session_state['authenticated'] = False
    st.rerun()

st.sidebar.divider()
if st.sidebar.button("🖥️ 바탕화면 바로가기 만들기", use_container_width=True):
    try:
        ok = create_desktop_shortcut()
        if ok:
            st.sidebar.success("✅ 바탕화면에 바로가기 생성 완료!")
        else:
            st.sidebar.error("생성 실패 — 바탕화면 경로를 확인해주세요.")
    except Exception as e:
        st.sidebar.error(f"오류: {e}")
st.sidebar.divider()

menu = st.sidebar.radio("작업 단계", [
    "📝 1단계: 대본 연구소",
    "⚙️ 2단계: 제작 스튜디오",
    "✅ 3단계: 최종 결과물"
])

# ════════════════════════════════════════════
# 1단계
# ════════════════════════════════════════════
if menu == "📝 1단계: 대본 연구소":
    st.title("📝 유튜브 알고리즘 대본 연구소")
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("콘텐츠 주제", placeholder="예: 100만 유튜버만 아는 편집 기술")
        style = st.selectbox("영상 스타일", ["정보 전달", "스토리텔링", "바이럴 뉴스", "자기계발"])
    with col2:
        dur_opt = st.select_slider("희망 분량", options=["숏폼 (60초)", "미드폼 (3분)", "롱폼 (8분)"])
        target = st.multiselect("주 시청층", ["2030","4050","전연령"], default=["전연령"])

    if st.button("🚀 AI 유튜브 대본 생성", use_container_width=True):
        with st.spinner("대본 작성 중..."):
            seed = random.randint(1000, 9999)
            seo_system = (
                "당신은 대한민국 최고의 유튜브 콘텐츠 전략가이자 대본 작가입니다. "
                "SEO 최적화, 알고리즘 친화적 구성, 시청자 후킹 기법을 완벽히 구사합니다. "
                "매번 독창적이고 신선한 각도의 대본을 작성하며 절대 이전과 같은 패턴을 반복하지 않습니다. "
                "대본에는 무대 지시사항(나레이션 톤, 화면 설명, [초] 타임라인)을 절대 포함하지 않습니다. "
                "오직 실제 말할 내용만 구어체로 작성합니다."
            )
            p = (
                f"[고유 시드: {seed}]\n"
                f"주제: '{topic}'\n스타일: {style}\n분량: {dur_opt}\n타겟: {target}\n\n"
                f"다음 조건을 모두 반영해서 유튜브 대본을 작성해줘:\n"
                f"1. 첫 10초 안에 강력한 후킹 (시청자가 끝까지 볼 이유)\n"
                f"2. SEO 핵심 키워드 자연스럽게 반복 포함\n"
                f"3. 유튜브 알고리즘이 좋아하는 구조 (문제 제기 → 핵심 내용 → 행동 유도)\n"
                f"4. 구어체, 자연스러운 호흡 (읽을 때 막힘 없게)\n"
                f"5. 대본만 출력 — 무대 지시사항, 나레이션 톤 표시, 화면 설명, 시간 마커 절대 금지"
            )
            res, engine = ask_ai(p, system_instruction=seo_system)
            st.session_state['script'] = res
            st.session_state['topic'] = topic
            st.success(f"완료! (엔진: {engine})")

        with st.spinner("BGM 분석 중..."):
            bgm_res, _ = ask_ai(
                f"대본 분위기에 맞는 유튜브 무료 BGM 검색어 5개 (한국어+영어, 번호 포함):\n\n{res[:400]}",
                "유튜브 BGM 전문 큐레이터"
            )
            st.session_state['bgm_recommendations'] = bgm_res

    if st.session_state['bgm_recommendations']:
        st.divider()
        st.subheader("🎵 추천 BGM 검색어")
        st.info(st.session_state['bgm_recommendations'])

    st.session_state['script'] = st.text_area("대본 수정", st.session_state['script'], height=350)

# ════════════════════════════════════════════
# 2단계
# ════════════════════════════════════════════
elif menu == "⚙️ 2단계: 제작 스튜디오":
    st.title("⚙️ AI 제작 스튜디오")
    if st.session_state['script']:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎙️ 음성 생성", "📄 자막 (.srt)", "🖼️ 썸네일 생성", "🎬 영상 자동 생성"
        ])

        # ── 탭1: 음성 ──
        with tab1:
            st.info("✅ 마크다운(*, #, 이모지 등) 자동 제거 후 음성 생성")
            v_map = {
                "남성 - 중후함 (InJoon)": "ko-KR-InJoonNeural",
                "여성 - 밝음 (SunHi)": "ko-KR-SunHiNeural",
                "여성 - 차분함 (HyunSu)": "ko-KR-HyunSuNeural",
                "남성 - 깔끔함 (BongJin)": "ko-KR-BongJinNeural"
            }
            selected_v = st.selectbox("목소리", list(v_map.keys()))
            col_r, col_p = st.columns(2)
            with col_r:
                rate = st.select_slider("말하기 속도", options=["-20%","-15%","-10%","-5%","0%","+5%","+10%"], value="-5%")
            with col_p:
                pitch = st.select_slider("음높이", options=["-10Hz","-5Hz","+0Hz","+5Hz","+10Hz"], value="+0Hz")
            if st.button("🎙️ 음성 생성", use_container_width=True):
                with st.spinner("음성 생성 중..."):
                    try:
                        create_voice_sync(st.session_state['script'][:2000], v_map[selected_v], rate, pitch)
                        st.audio("output/voice.mp3")
                        st.success("✅ 완료!")
                    except Exception as e:
                        st.error(f"음성 생성 오류: {e}")

        # ── 탭2: 자막 ──
        with tab2:
            chars_per_sec = st.slider("발화 속도 (초당 글자수)", 2.0, 7.0, 4.5, 0.5)
            if st.button("📄 자막 생성"):
                srt = generate_srt(st.session_state['script'], chars_per_sec)
                with open("output/subtitle.srt", "w", encoding="utf-8") as f:
                    f.write(srt)
                st.success("✅ 완료!")
                st.text_area("미리보기", srt, height=300)
                with open("output/subtitle.srt", "rb") as f:
                    st.download_button("⬇️ .srt 다운로드", f, "subtitle.srt", mime="text/plain")

        # ── 탭3: 썸네일 ──
        with tab3:
            st.subheader("📺 썸네일 문구 + AI 이미지 생성")

            # 문구 분석
            if st.button("✨ AI 썸네일 문구 분석 (나노바나나)", use_container_width=True):
                with st.spinner("나노바나나가 분석 중..."):
                    res, engine = ask_visual_ai(
                        f"대본: {st.session_state['script'][:500]}\n\n"
                        f"1. 조회수를 폭발시킬 짧은 썸네일 문구 3개 (10자 이내, 임팩트 있게)\n"
                        f"2. 썸네일 배경 AI 이미지 영문 프롬프트 1개 (cinematic, 16:9 YouTube thumbnail style)"
                    )
                    st.session_state['thumb_text'] = res
                    # 프롬프트만 추출
                    lines = [l.strip() for l in res.split('\n') if l.strip()]
                    for l in reversed(lines):
                        if any(w in l.lower() for w in ['cinematic','thumbnail','background','style','4k','dramatic']):
                            st.session_state['thumb_img_prompt'] = re.sub(r'^[\d\.\-\*\s]+', '', l)
                            break

            if st.session_state['thumb_text']:
                st.info(st.session_state['thumb_text'])

            st.divider()
            st.subheader("🖼️ 썸네일 이미지 직접 생성")

            col_tp, col_ts = st.columns([3, 1])
            with col_tp:
                thumb_prompt = st.text_input(
                    "이미지 프롬프트 (영문)",
                    value=st.session_state.get('thumb_img_prompt', ''),
                    placeholder="예: dramatic cinematic YouTube thumbnail, dark background, neon"
                )
            with col_ts:
                thumb_size = st.selectbox("사이즈", ["1280x720 (16:9)", "1080x1080 (1:1)"], index=0)

            tw, th = (1280, 720) if "1280" in thumb_size else (1080, 1080)

            if st.button("🎨 썸네일 이미지 생성", use_container_width=True, type="primary"):
                if not thumb_prompt:
                    st.warning("프롬프트를 입력하거나 위에서 AI 분석을 먼저 해주세요.")
                else:
                    with st.spinner("Pollinations.ai로 이미지 생성 중... (약 10~20초)"):
                        img = fetch_ai_bg(thumb_prompt, tw, th, seed=999)
                        if img:
                            img_path = "output/thumbnail.png"
                            img.save(img_path)
                            st.image(img, caption="생성된 썸네일", use_container_width=True)
                            with open(img_path, "rb") as f:
                                st.download_button("⬇️ 썸네일 다운로드 (PNG)", f, "thumbnail.png", mime="image/png")
                            st.success("✅ 썸네일 생성 완료!")
                        else:
                            st.error("이미지 생성 실패. 프롬프트를 바꿔서 다시 시도해 보세요.")

            st.divider()
            if st.button("🎬 Pika/Runway용 영문 프롬프트"):
                res, _ = ask_visual_ai(
                    f"대본: {st.session_state['script'][:300]}\n"
                    f"기반의 Pika/Runway 영상 생성 영문 프롬프트 3개 (각각 cinematic style, 다른 분위기로)"
                )
                st.code(res)

        # ── 탭4: 영상 생성 ──
        with tab4:
            st.subheader("🎬 자동 영상 생성")
            st.caption("나노바나나 씬 기획 · Ken Burns 줌 · AI 배경 · YouTube 자막 스타일")

            col_a, col_b = st.columns(2)
            with col_a:
                use_ai_bg = st.checkbox("🖼️ AI 배경 (Pollinations.ai)", value=True)
                num_bgs = st.slider("배경 이미지 수", 2, 8, 4)
            with col_b:
                bg_override = st.text_input(
                    "배경 프롬프트 직접 입력 (비우면 나노바나나 자동 생성)",
                    value="",
                    placeholder="비우면 나노바나나가 자동으로 씬 기획"
                )

            if not os.path.exists("output/voice.mp3"):
                st.warning("⚠️ 먼저 '음성 생성' 탭에서 음성을 만들어 주세요.")
            else:
                st.info("나노바나나가 씬별 배경 프롬프트를 자동 기획합니다.")
                if st.button("🚀 영상 생성 시작", use_container_width=True, type="primary"):
                    progress_bar = st.progress(0)
                    status_txt = st.empty()

                    def cb(val, msg):
                        progress_bar.progress(min(float(val), 1.0))
                        status_txt.text(msg)

                    try:
                        out = create_auto_video(
                            script=st.session_state['script'],
                            topic=st.session_state.get('topic', ''),
                            use_ai_bg=use_ai_bg,
                            bg_prompt=bg_override,
                            num_ai_bgs=num_bgs,
                            progress_cb=cb
                        )
                        st.success("✅ 영상 생성 완료!")
                        st.video(out)
                        with open(out, "rb") as f:
                            st.download_button("⬇️ 영상 다운로드", f, "auto_video.mp4")
                    except Exception as e:
                        st.error(f"오류: {e}")
    else:
        st.warning("대본을 먼저 생성해 주세요.")

# ════════════════════════════════════════════
# 3단계
# ════════════════════════════════════════════
else:
    st.title("✅ 최종 제작물")

    if os.path.exists("output/thumbnail.png"):
        st.subheader("🖼️ 썸네일")
        st.image("output/thumbnail.png", use_container_width=True)
        with open("output/thumbnail.png", "rb") as f:
            st.download_button("📂 썸네일 다운로드", f, "thumbnail.png", mime="image/png")
        st.divider()

    if os.path.exists("output/auto_video.mp4"):
        st.subheader("🎬 자동 생성 영상")
        st.video("output/auto_video.mp4")
        with open("output/auto_video.mp4", "rb") as f:
            st.download_button("📂 영상 다운로드", f, "auto_video.mp4")
        st.divider()

    st.subheader("📤 외부 영상 업로드 (Pika/Runway)")
    uploaded = st.file_uploader("영상 파일", type=["mp4","mov","avi","webm"])
    if uploaded:
        vpath = f"output/uploaded_{uploaded.name}"
        with open(vpath, "wb") as f:
            f.write(uploaded.read())
        st.video(vpath)
        with open(vpath, "rb") as f:
            st.download_button("📂 다운로드", f, uploaded.name)
    st.divider()

    if os.path.exists("output/voice.mp3"):
        st.subheader("🎙️ 음성")
        st.audio("output/voice.mp3")
        with open("output/voice.mp3", "rb") as f:
            st.download_button("📂 음성 다운로드", f, "youtube_voice.mp3")

    if os.path.exists("output/subtitle.srt"):
        st.subheader("📄 자막")
        with open("output/subtitle.srt", "r", encoding="utf-8") as f:
            srt_data = f.read()
        st.text_area("자막 내용", srt_data, height=200)
        with open("output/subtitle.srt", "rb") as f:
            st.download_button("📂 자막 다운로드", f, "subtitle.srt")

    if not any(os.path.exists(p) for p in [
        "output/voice.mp3","output/subtitle.srt",
        "output/auto_video.mp4","output/thumbnail.png"]):
        st.info("아직 결과물이 없습니다. 1단계부터 시작해 주세요.")
