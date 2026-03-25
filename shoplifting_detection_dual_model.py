"""
Shoplifting Detection — GUI + Gmail Notifications  [VERSION DOUBLE MODÈLE]
══════════════════════════════════════════════════════════════════════════════
Architecture pipeline à 2 modèles :
  • Modèle 1 : YOLOv8n (COCO) — détection d'objets (personnes, sacs, bouteilles…)
  • Modèle 2 : shoplifting_wights.pt — analyse comportementale (VOL / NORMAL)
  Les deux tournent en parallèle via ThreadPoolExecutor pour minimiser la latence.

Autres fonctionnalités (inchangées) :
  • Interface graphique tkinter (dark, épurée)
  • Drag-and-drop de fichier vidéo (nécessite tkinterdnd2)
  • Bouton Webcam  (index 0 par défaut)
  • Prévisualisation live dans la fenêtre
  • Envoi d'email Gmail (SMTP SSL) avec snapshot en pièce jointe
  • Cooldown configurable entre deux emails
  • Compteur d'alertes + journal en temps réel
  • Bouton « Tester l'envoi »
"""

from ultralytics import YOLO
import numpy as np
import cv2
import threading
import concurrent.futures
import queue
import time
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# ══════════════════════════════════════════════════════════════════════════════
#  CHEMINS DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH        = r"c:\Users\divhi\OneDrive\Documents\UCAC-ICAM\X2_2025-2026\PRESENTATION JPO\shoplifting_wights.pt"

OBJECT_MODEL_PATH = r"c:\Users\divhi\OneDrive\Documents\UCAC-ICAM\X2_2025-2026\PRESENTATION JPO\yolo11n.pt"

OUTPUT_DIR        = r"c:\Users\divhi\OneDrive\Documents\UCAC-ICAM\X2_2025-2026\PRESENTATION JPO\build"

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES VIDÉO / INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════
DISPLAY_WIDTH     = 720
INFER_WIDTH       = 416
FRAME_SKIP        = 2
CONF_THRESHOLD    = 0.45    
NMS_IOU           = 0.45
SMOOTHING_WINDOW  = 10
SHOPLIFTING_RATIO = 0.4

OBJECT_CONF       = 0.50
# Classes COCO conservées : person=0, backpack=24, handbag=26,
#                           suitcase=28, bottle=39, cup=41, cell phone=67
OBJECT_CLASSES    = [0, 24, 26, 28, 39, 41, 67]
COCO_LABELS       = {
    0:  "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    67: "phone",
}
# Couleurs pour les boîtes objets (BGR) — bleu clair, distinct du rouge VOL
COLOR_OBJECT_BOX  = (255, 180,  60)   # orange clair
COLOR_OBJECT_TEXT = (255, 180,  60)

EMAIL_COOLDOWN    = 60      # secondes entre deux emails

# ── Couleurs annotations ──────────────────────────────────────────────────────
COLOR_SHOPLIFTING  = (0,   0,   255)
COLOR_NORMAL       = (0,   200,   0)
COLOR_CONF_TEXT    = (255, 255,   0)
COLOR_STATUS_ALERT = (0,   0,   255)
COLOR_STATUS_OK    = (0,   200,   0)
COLOR_FPS          = (200, 200, 200)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSES UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

class VideoStream:
    """Lit les frames dans un thread séparé pour ne pas bloquer l'inférence."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"[ERREUR] Impossible d'ouvrir : {src}")
        self.q       = queue.Queue(maxsize=4)
        self.stopped = False
        self.thread  = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        return self.q.get(timeout=2)

    def more(self):
        return not (self.stopped and self.q.empty())

    def stop(self):
        self.stopped = True
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def resize_for_display(frame, width):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)),
                      interpolation=cv2.INTER_LINEAR)


def resize_for_inference(frame, width):
    h, w  = frame.shape[:2]
    ratio = width / w
    small = cv2.resize(frame, (width, int(h * ratio)),
                       interpolation=cv2.INTER_LINEAR)
    return small, ratio


def draw_behavior_box(frame, x1, y1, w, h, conf, clas):
    """Dessine la boîte comportementale (VOL / OK) — logique originale inchangée."""
    color     = COLOR_SHOPLIFTING if clas == 1 else COLOR_NORMAL
    thickness = 2 if clas == 1 else 1
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, thickness)
    label = f"{'VOL' if clas == 1 else 'OK'} {conf*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if clas == 1:
        cx = x1 + w // 2
        cv2.circle(frame, (cx, y1), 5, COLOR_SHOPLIFTING, -1)


def draw_object_box(frame, x1, y1, x2, y2, label_text):
    """
    Dessine la boîte objet (Modèle 1 — COCO).
    Style différent (contour fin, coin marqué) pour ne pas confondre avec VOL.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBJECT_BOX, 1)
    # Petits coins distinctifs pour différencier visuellement
    ln = 10
    pts = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts, dirs):
        cv2.line(frame, (px, py), (px + dx*ln, py), COLOR_OBJECT_BOX, 2)
        cv2.line(frame, (px, py), (px, py + dy*ln), COLOR_OBJECT_BOX, 2)
    # Label
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 4, y2 + th + 5),
                  COLOR_OBJECT_BOX, -1)
    cv2.putText(frame, label_text, (x1 + 2, y2 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)


def draw_status_bar(frame, status_text, fps, is_alert, obj_count=0):
    """Barre de statut avec FPS + compteur d'objets détectés."""
    color = COLOR_STATUS_ALERT if is_alert else COLOR_STATUS_OK
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FPS, 1)
    # Nouveau : affiche le nombre d'objets détectés par le Modèle 1
    cv2.putText(frame, f"Objets: {obj_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 100), 1)


def _is_alert(history: deque) -> bool:
    if not history:
        return False
    return sum(history) / len(history) >= SHOPLIFTING_RATIO


def _current_status(history: deque) -> str:
    return "! VOL DETECTE!" if _is_alert(history) else "OK Comportement normal"


def _write_frame(writer, frame):
    if writer is not None:
        writer.write(frame)


# ══════════════════════════════════════════════════════════════════════════════
#  EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def send_alert_email(sender_email: str, sender_password: str,
                     recipient_email: str, snapshot=None):
    """
    Envoie une alerte via Gmail SMTP SSL.
    Retourne True si succès, sinon le message d'erreur (str).
    """
    try:
        msg            = MIMEMultipart()
        msg["From"]    = sender_email
        msg["To"]      = recipient_email
        msg["Subject"] = (
            f"ALERTE VOL DETECTE -- "
            f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )

        body = (
            f"Une activite suspecte a ete detectee par le systeme de surveillance.\n\n"
            f"Date/Heure : {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')}\n"
            f"Source     : Camera de surveillance\n\n"
            f"Veuillez verifier les enregistrements immediatement.\n\n"
            f"--- Systeme de detection automatique"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if snapshot is not None:
            _, buf      = cv2.imencode(".jpg", snapshot,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_data    = buf.tobytes()
            attachment  = MIMEImage(img_data, name="alerte_snapshot.jpg")
            attachment.add_header("Content-Disposition", "attachment",
                                  filename="alerte_snapshot.jpg")
            msg.attach(attachment)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as srv:
            srv.login(sender_email, sender_password)
            srv.sendmail(sender_email, recipient_email, msg.as_string())
        return True
    except Exception as exc:
        return str(exc)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class ShopliftingApp:

    BG_DEEP   = "#0d0d18"
    BG_PANEL  = "#13131f"
    BG_CARD   = "#1a1a2e"
    BG_INPUT  = "#0f0f1a"
    ACCENT    = "#e94560"
    ACCENT2   = "#0f3460"
    FG_MAIN   = "#e8e8f0"
    FG_DIM    = "#5a6070"
    FG_OK     = "#3ddc84"
    FONT_UI   = ("Segoe UI", 9)
    FONT_BOLD = ("Segoe UI", 9, "bold")
    FONT_MONO = ("Consolas",  8)

    def __init__(self, root: tk.Tk):
        self.root           = root
        self.source_path    = None
        self.running        = False

        # ── Modèles (initialisés à None, chargés en async) ───────────────────
        self.model          = None   # Modèle 2 — comportemental (existant)
        self.object_model   = None   # Modèle 1 — détection d'objets (nouveau)

        self._det_thread    = None
        self._last_email    = 0.0
        self._alert_count   = 0
        self._fq            = queue.Queue(maxsize=2)
        self._photo_ref     = None

        # ThreadPoolExecutor réutilisé pour les inférences parallèles
        self._infer_pool    = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self._setup_window()
        self._build_ui()
        self._load_models_async()   # ← charge les 2 modèles en parallèle

    # ── Fenêtre ───────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("Shoplifting Detection  •  AI Surveillance")
        self.root.configure(bg=self.BG_DEEP)
        self.root.minsize(1020, 640)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww, wh = 1080, 720
        self.root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")

    # ── Interface ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg=self.BG_PANEL, pady=0)
        top.pack(fill="x")
        tk.Frame(top, bg=self.ACCENT, height=3).pack(fill="x")
        inner_top = tk.Frame(top, bg=self.BG_PANEL, pady=10)
        inner_top.pack(fill="x", padx=16)

        tk.Label(inner_top,
                 text="SHOPLIFTING APPLICATION- SUSPECT",
                 font=("Courier New", 13, "bold"),
                 bg=self.BG_PANEL, fg=self.ACCENT).pack(side="left")

        self._model_lbl = tk.Label(inner_top,
                                   text="⏳  Chargement des modèles…",
                                   font=self.FONT_UI,
                                   bg=self.BG_PANEL, fg=self.FG_DIM)
        self._model_lbl.pack(side="right")

        body = tk.Frame(self.root, bg=self.BG_DEEP)
        body.pack(fill="both", expand=True, padx=14, pady=(10, 14))

        left = tk.Frame(body, bg=self.BG_DEEP)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left,
                                width=DISPLAY_WIDTH, height=405,
                                bg=self.BG_INPUT,
                                highlightthickness=1,
                                highlightbackground="#2a2a40")
        self.canvas.pack()
        self._show_placeholder()

        dz = tk.Frame(left, bg=self.BG_DEEP)
        dz.pack(fill="x", pady=(8, 0))

        hint = ("📂  Glisser-deposer une video ici  —  ou  cliquer pour parcourir"
                if HAS_DND else
                "📂  Cliquer pour choisir une video")
        self._drop_lbl = tk.Label(
            dz, text=hint,
            font=("Segoe UI", 9, "italic"),
            bg="#12121e", fg=self.FG_DIM,
            pady=14, padx=10, cursor="hand2",
            relief="flat"
        )
        self._drop_lbl.pack(fill="x")
        tk.Frame(dz, bg=self.ACCENT2, height=1).pack(fill="x")

        if HAS_DND:
            self._drop_lbl.drop_target_register(DND_FILES)
            self._drop_lbl.dnd_bind("<<Drop>>", self._on_drop)
        self._drop_lbl.bind("<Button-1>", lambda _: self._browse_file())

        btns = tk.Frame(left, bg=self.BG_DEEP)
        btns.pack(fill="x", pady=8)

        self._make_btn(btns, "📷  Webcam",    self.ACCENT2, self._use_webcam).pack(side="left", padx=(0, 6))
        self._start_btn = self._make_btn(btns, "▶  Demarrer", self.ACCENT, self._toggle)
        self._start_btn.pack(side="left")

        self._status_var = tk.StringVar(value="Prêt — choisissez une source vidéo")
        tk.Label(left, textvariable=self._status_var,
                 font=("Segoe UI", 8, "italic"),
                 bg=self.BG_DEEP, fg=self.FG_DIM,
                 anchor="w").pack(fill="x", pady=(2, 0))

        right = tk.Frame(body, bg=self.BG_DEEP, width=290)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        self._build_card(right, self._card_models)
        self._build_card(right, self._card_alerts)
        self._build_card(right, self._card_email)
        self._build_card(right, self._card_log, expand=True)

    def _make_btn(self, parent, text, bg, cmd):
        return tk.Button(parent, text=text,
                         font=self.FONT_BOLD,
                         bg=bg, fg=self.FG_MAIN,
                         activebackground=bg, activeforeground=self.FG_MAIN,
                         relief="flat", padx=14, pady=7,
                         cursor="hand2", command=cmd, bd=0)

    def _build_card(self, parent, builder_fn, expand=False):
        card = tk.Frame(parent, bg=self.BG_CARD, padx=14, pady=12, bd=0)
        if expand:
            card.pack(fill="both", expand=True)
        else:
            card.pack(fill="x", pady=(0, 10))
        builder_fn(card)

    def _lbl(self, parent, text):
        tk.Label(parent, text=text,
                 font=("Segoe UI", 7, "bold"),
                 bg=self.BG_CARD, fg=self.FG_DIM).pack(anchor="w", pady=(6, 1))

    def _entry(self, parent, var, show=None):
        kw = dict(show=show) if show else {}
        e  = tk.Entry(parent, textvariable=var,
                      font=self.FONT_UI,
                      bg=self.BG_INPUT, fg=self.FG_MAIN,
                      insertbackground=self.FG_MAIN,
                      relief="flat", bd=0, **kw)
        e.pack(fill="x", ipady=5)
        tk.Frame(parent, bg=self.ACCENT2, height=1).pack(fill="x")
        return e

    # ── Carte : statut des 2 modèles (NOUVEAU) ────────────────────────────────
    def _card_models(self, card):
        tk.Label(card, text="STATUT DES MODÈLES",
                 font=("Courier New", 7, "bold"),
                 bg=self.BG_CARD, fg=self.FG_DIM).pack(anchor="w", pady=(0, 4))

        row1 = tk.Frame(card, bg=self.BG_CARD)
        row1.pack(fill="x", pady=1)
        tk.Label(row1, text="Modèle 1 (objets) :",
                 font=self.FONT_UI, bg=self.BG_CARD, fg=self.FG_DIM,
                 width=20, anchor="w").pack(side="left")
        self._obj_model_lbl = tk.Label(row1, text="⏳ chargement…",
                                       font=self.FONT_UI,
                                       bg=self.BG_CARD, fg="#ffcc44")
        self._obj_model_lbl.pack(side="left")

        row2 = tk.Frame(card, bg=self.BG_CARD)
        row2.pack(fill="x", pady=1)
        tk.Label(row2, text="Modèle 2 (vol) :",
                 font=self.FONT_UI, bg=self.BG_CARD, fg=self.FG_DIM,
                 width=20, anchor="w").pack(side="left")
        self._bhv_model_lbl = tk.Label(row2, text="⏳ chargement…",
                                       font=self.FONT_UI,
                                       bg=self.BG_CARD, fg="#ffcc44")
        self._bhv_model_lbl.pack(side="left")

    def _card_alerts(self, card):
        tk.Label(card, text="ALERTES DETECTEES",
                 font=("Courier New", 7, "bold"),
                 bg=self.BG_CARD, fg=self.FG_DIM).pack(anchor="w")
        self._alert_var = tk.StringVar(value="0")
        tk.Label(card, textvariable=self._alert_var,
                 font=("Courier New", 40, "bold"),
                 bg=self.BG_CARD, fg=self.ACCENT).pack(anchor="w")
        self._alert_status = tk.Label(card, text="Aucune alerte",
                                      font=("Segoe UI", 8),
                                      bg=self.BG_CARD, fg=self.FG_OK)
        self._alert_status.pack(anchor="w")

    def _card_email(self, card):
        tk.Label(card, text="NOTIFICATIONS GMAIL",
                 font=("Courier New", 7, "bold"),
                 bg=self.BG_CARD, fg=self.FG_DIM).pack(anchor="w", pady=(0, 4))

        self._lbl(card, "EMAIL EXPEDITEUR (Gmail)")
        self._sender_var = tk.StringVar()
        self._entry(card, self._sender_var)

        self._lbl(card, "MOT DE PASSE APP GOOGLE")
        self._pwd_var = tk.StringVar()
        self._entry(card, self._pwd_var, show="*")

        self._lbl(card, "EMAIL DESTINATAIRE")
        self._recip_var = tk.StringVar()
        self._entry(card, self._recip_var)

        self._lbl(card, "DELAI ENTRE EMAILS (sec)")
        self._cooldown_var = tk.IntVar(value=EMAIL_COOLDOWN)
        sp = tk.Spinbox(card, from_=10, to=3600,
                        textvariable=self._cooldown_var,
                        font=self.FONT_UI,
                        bg=self.BG_INPUT, fg=self.FG_MAIN,
                        buttonbackground=self.BG_CARD,
                        insertbackground=self.FG_MAIN,
                        relief="flat", bd=0)
        sp.pack(fill="x", ipady=4)
        tk.Frame(card, bg=self.ACCENT2, height=1).pack(fill="x")

        self._make_btn(card, "✉  Tester l'envoi", self.ACCENT2,
                       self._test_email).pack(anchor="w", pady=(10, 0))

    def _card_log(self, card):
        tk.Label(card, text="JOURNAL",
                 font=("Courier New", 7, "bold"),
                 bg=self.BG_CARD, fg=self.FG_DIM).pack(anchor="w", pady=(0, 6))
        self._log_box = scrolledtext.ScrolledText(
            card,
            font=self.FONT_MONO,
            bg=self.BG_INPUT, fg="#7a8fa6",
            insertbackground=self.FG_DIM,
            relief="flat", bd=0,
            wrap="word", state="disabled"
        )
        self._log_box.pack(fill="both", expand=True)

    def _show_placeholder(self):
        self.canvas.delete("all")
        cx = DISPLAY_WIDTH // 2
        self.canvas.create_text(cx, 180, text="▶",
                                fill="#1e1e30", font=("Courier New", 72))
        self.canvas.create_text(cx, 300,
                                text="Glisser une video  /  Webcam",
                                fill="#2a2a45", font=("Courier New", 12))

    def _log(self, msg: str):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        self.root.after(0, self._append_log, line)

    def _append_log(self, line: str):
        self._log_box.config(state="normal")
        self._log_box.insert("end", line)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    # ── Chargement des 2 modèles en parallèle (MODIFIÉ) ──────────────────────
    def _load_models_async(self):
        def _load_behavior():
            """Charge le modèle comportemental (Modèle 2)."""
            try:
                m = YOLO(MODEL_PATH)
                m.to("cpu")
                self.model = m
                self.root.after(0, lambda: self._bhv_model_lbl.config(
                    text="✅ prêt", fg=self.FG_OK))
                self._log("Modèle 2 (comportemental) chargé.")
            except Exception as exc:
                self.root.after(0, lambda: self._bhv_model_lbl.config(
                    text=f"❌ {exc}", fg=self.ACCENT))
                self._log(f"Erreur Modèle 2 : {exc}")

        def _load_objects():
            """Charge le modèle de détection d'objets (Modèle 1 — YOLOv8n COCO)."""
            try:
                m = YOLO(OBJECT_MODEL_PATH)   # auto-télécharge yolov8n.pt si absent
                m.to("cpu")
                self.object_model = m
                self.root.after(0, lambda: self._obj_model_lbl.config(
                    text="✅ prêt", fg=self.FG_OK))
                self._log("Modèle 1 (détection objets YOLOv8n) chargé.")
            except Exception as exc:
                self.root.after(0, lambda: self._obj_model_lbl.config(
                    text=f"❌ {exc}", fg=self.ACCENT))
                self._log(f"Erreur Modèle 1 : {exc}")

        def _run():
            # Lance les 2 chargements en parallèle
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_load_objects)
                f2 = pool.submit(_load_behavior)
                concurrent.futures.wait([f1, f2])
            # Met à jour le label global quand tout est prêt
            if self.model is not None and self.object_model is not None:
                self.root.after(0, lambda: self._model_lbl.config(
                    text="✅  2 modèles prêts", fg=self.FG_OK))
            else:
                self.root.after(0, lambda: self._model_lbl.config(
                    text="⚠️  Un modèle en erreur", fg="#ffcc44"))

        threading.Thread(target=_run, daemon=True).start()

    # ── Sélection de la source ────────────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Choisir une vidéo",
            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                       ("Tous fichiers", "*.*")]
        )
        if path:
            self._set_source(path)

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        if os.path.isfile(path):
            self._set_source(path)

    def _use_webcam(self):
        self._set_source(0)

    def _set_source(self, path):
        if self.running:
            self._stop()
        self.source_path = path
        name = "Webcam (index 0)" if path == 0 else os.path.basename(str(path))
        self._status_var.set(f"Source : {name}")
        self._drop_lbl.config(text=f"✅  {name}", fg=self.FG_OK)
        self._log(f"Source : {name}")

    # ── Démarrer / Arrêter ────────────────────────────────────────────────────
    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if self.model is None or self.object_model is None:
            messagebox.showwarning("Modèles",
                "Les modèles ne sont pas encore chargés.\n"
                "Attendez que les deux soient marqués ✅ prêt.")
            return
        if self.source_path is None:
            messagebox.showwarning("Source", "Choisissez d'abord une source vidéo.")
            return
        self.running      = True
        self._alert_count = 0
        self._alert_var.set("0")
        self._alert_status.config(text="Surveillance en cours…", fg=self.FG_OK)
        self._start_btn.config(text="⏹  Arrêter", bg="#333345")
        self._log("Détection démarrée (pipeline double modèle).")
        self._det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._det_thread.start()
        self._pump_canvas()

    def _stop(self):
        self.running = False
        self._start_btn.config(text="▶  Demarrer", bg=self.ACCENT)
        self._status_var.set("Arrêté.")
        self._log("Détection arrêtée.")
        self.root.after(300, self._show_placeholder)

    # ══════════════════════════════════════════════════════════════════════════
    #  BOUCLE DE DÉTECTION — PIPELINE DOUBLE MODÈLE
    # ══════════════════════════════════════════════════════════════════════════
    def _detection_loop(self):
        src = self.source_path if self.source_path != 0 else 0
        try:
            vs = VideoStream(src).start()
        except IOError as exc:
            self._log(f"Erreur ouverture source : {exc}")
            self.root.after(0, self._stop)
            return

        time.sleep(0.5)   # laisse le buffer se remplir

        writer         = None
        history: deque = deque(maxlen=SMOOTHING_WINDOW)
        frame_count    = 0
        last_result    = None
        fps_timer      = time.time()
        fps            = 0.0
        fps_frames     = 0
        prev_alert     = False

        # Fonction d'inférence Modèle 1 — capture les variables ici
        def _run_object_model(frame):
            return self.object_model.predict(
                frame,
                conf    = OBJECT_CONF,
                classes = OBJECT_CLASSES,
                iou     = NMS_IOU,
                device  = "cpu",
                imgsz   = INFER_WIDTH,
                verbose = False,
            )

        # Fonction d'inférence Modèle 2 — identique à l'original
        def _run_behavior_model(frame):
            return self.model.predict(
                frame,
                conf    = CONF_THRESHOLD,
                iou     = NMS_IOU,
                device  = "cpu",
                imgsz   = INFER_WIDTH,
                verbose = False,
                half    = False,
                augment = False,
            )

        while self.running and vs.more():
            try:
                raw_frame = vs.read()
            except queue.Empty:
                break

            frame_count += 1
            fps_frames  += 1

            # ── Calcul FPS ────────────────────────────────────────────────────
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps        = fps_frames / elapsed
                fps_frames = 0
                fps_timer  = time.time()

            # ── Redimensionner pour affichage ─────────────────────────────────
            display_frame = resize_for_display(raw_frame, DISPLAY_WIDTH)

            # ── Saut de frames (FRAME_SKIP) ───────────────────────────────────
            if frame_count % FRAME_SKIP != 0:
                show = (display_frame.copy()
                        if last_result is None else last_result.copy())
                draw_status_bar(show, _current_status(history), fps,
                                _is_alert(history))
                self._push_frame(show)
                _write_frame(writer, show)
                continue

            # ── Image d'inférence (résolution réduite) ────────────────────────
            infer_frame, _ = resize_for_inference(raw_frame, INFER_WIDTH)
            s2d            = DISPLAY_WIDTH / INFER_WIDTH   # scale inférence → affichage

            # ── Inférences parallèles (Modèle 1 + Modèle 2) ──────────────────
            future_obj = self._infer_pool.submit(_run_object_model, infer_frame)
            future_bhv = self._infer_pool.submit(_run_behavior_model, infer_frame)
            results_obj = future_obj.result()
            results_bhv = future_bhv.result()

            # ── Annotation ────────────────────────────────────────────────────
            annotated            = display_frame.copy()
            shoplifting_in_frame = False
            obj_count            = 0

            # — Modèle 1 : boîtes objets (couche visuelle basse) —
            boxes_obj = results_obj[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy_obj  = boxes_obj.xyxy.cpu().numpy()
                confs_obj = boxes_obj.conf.cpu().numpy()
                clses_obj = boxes_obj.cls.cpu().numpy().astype(int)
                obj_count = len(xyxy_obj)

                for i in range(len(xyxy_obj)):
                    x1 = int(xyxy_obj[i][0] * s2d)
                    y1 = int(xyxy_obj[i][1] * s2d)
                    x2 = int(xyxy_obj[i][2] * s2d)
                    y2 = int(xyxy_obj[i][3] * s2d)
                    conf_o  = float(confs_obj[i])
                    cls_id  = int(clses_obj[i])
                    lbl     = COCO_LABELS.get(cls_id, str(cls_id))
                    draw_object_box(annotated, x1, y1, x2, y2,
                                    f"{lbl} {conf_o*100:.0f}%")

            # — Modèle 2 : boîtes comportementales (couche visuelle haute) —
            boxes_bhv = results_bhv[0].boxes
            if boxes_bhv is not None and len(boxes_bhv) > 0:
                xyxy_bhv  = boxes_bhv.xyxy.cpu().numpy()
                xywh_bhv  = boxes_bhv.xywh.cpu().numpy()
                confs_bhv = boxes_bhv.conf.cpu().numpy()
                clses_bhv = boxes_bhv.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy_bhv)):
                    conf = float(confs_bhv[i])
                    clas = int(clses_bhv[i])
                    x1   = int(xyxy_bhv[i][0] * s2d)
                    y1   = int(xyxy_bhv[i][1] * s2d)
                    w    = int(xywh_bhv[i][2] * s2d)
                    h    = int(xywh_bhv[i][3] * s2d)

                    # Filtre de confiance pour la classe "normal" (logique originale)
                    if clas == 0 and conf < 0.80:
                        continue

                    draw_behavior_box(annotated, x1, y1, w, h, conf, clas)
                    if clas == 1:
                        shoplifting_in_frame = True

            # ── Historique & alertes ───────────────────────────────────────────
            history.append(shoplifting_in_frame)
            is_alert    = _is_alert(history)
            status_text = _current_status(history)
            draw_status_bar(annotated, status_text, fps, is_alert, obj_count)
            last_result = annotated

            if is_alert and not prev_alert:
                self._alert_count += 1
                cnt = self._alert_count
                ts  = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, lambda c=cnt, t=ts: (
                    self._alert_var.set(str(c)),
                    self._alert_status.config(
                        text=f"Dernière alerte : {t}", fg=self.ACCENT)
                ))
                self._log(f"Alerte #{self._alert_count} declenchee !")
                self._maybe_send_email(annotated.copy())

            prev_alert = is_alert
            self._push_frame(annotated)

            # ── Initialisation du writer vidéo ────────────────────────────────
            if OUTPUT_DIR and writer is None and os.path.isdir(OUTPUT_DIR):
                h_out, w_out = annotated.shape[:2]
                fourcc       = cv2.VideoWriter_fourcc(*"MJPG")
                ts_str       = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path     = os.path.join(OUTPUT_DIR, f"detection_{ts_str}.avi")
                writer       = cv2.VideoWriter(out_path, fourcc, 25, (w_out, h_out))
                self._log(f"Enregistrement → {out_path}")

            _write_frame(writer, annotated)

        vs.stop()
        if writer:
            writer.release()
        self.root.after(0, self._stop)

    # ── Canvas ────────────────────────────────────────────────────────────────
    def _push_frame(self, frame):
        try:
            self._fq.put_nowait(frame)
        except queue.Full:
            pass

    def _pump_canvas(self):
        if not self.running:
            return
        try:
            frame = self._fq.get_nowait()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            cw    = self.canvas.winfo_width()  or DISPLAY_WIDTH
            ch    = self.canvas.winfo_height() or 405
            pil   = pil.resize((cw, ch), Image.BILINEAR)
            self._photo_ref = ImageTk.PhotoImage(pil)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo_ref)
        except queue.Empty:
            pass
        self.root.after(16, self._pump_canvas)   # ~60 Hz

    # ── Email ─────────────────────────────────────────────────────────────────
    def _maybe_send_email(self, snapshot):
        sender    = self._sender_var.get().strip()
        password  = self._pwd_var.get().strip()
        recipient = self._recip_var.get().strip()
        if not (sender and password and recipient):
            self._log("Email non configuré — alerte non envoyée.")
            return
        cooldown = self._cooldown_var.get()
        now      = time.time()
        if now - self._last_email < cooldown:
            left = int(cooldown - (now - self._last_email))
            self._log(f"Email ignoré (cooldown, {left}s restantes).")
            return
        self._last_email = now

        def _send():
            res = send_alert_email(sender, password, recipient, snapshot)
            if res is True:
                self._log(f"Email envoyé → {recipient}")
            else:
                self._log(f"Erreur email : {res}")

        threading.Thread(target=_send, daemon=True).start()

    def _test_email(self):
        sender    = self._sender_var.get().strip()
        password  = self._pwd_var.get().strip()
        recipient = self._recip_var.get().strip()
        if not (sender and password and recipient):
            messagebox.showwarning("Email", "Remplissez tous les champs email.")
            return

        def _send():
            res = send_alert_email(sender, password, recipient)
            if res is True:
                self.root.after(0, lambda: messagebox.showinfo(
                    "Email", f"Email de test envoyé à {recipient}"))
                self._log(f"Email de test → {recipient}")
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Erreur email", str(res)))
                self._log(f"Erreur email de test : {res}")

        threading.Thread(target=_send, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = ShopliftingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
