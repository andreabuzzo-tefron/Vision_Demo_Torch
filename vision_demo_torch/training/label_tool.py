from flask import Flask, render_template_string, request, redirect, url_for
from pathlib import Path
import csv

app = Flask(__name__)

# percorsi (adatta se hai cartelle diverse)
BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = (BASE_DIR.parent / "app" / "data_capture").resolve()
CSV_PATH = (BASE_DIR / "dataset_labeled.csv").resolve()

# alfabeto di riferimento (puoi cambiare)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"

# template HTML inline per semplicità
PAGE_TMPL = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Label OCR</title>
    <style>
      body { font-family: sans-serif; margin: 20px; }
      .img-box { margin-bottom: 20px; }
      img { border: 1px solid #ccc; max-height: 250px; }
      input[type=text] { font-size: 1.2rem; padding: 4px 8px; }
      .btn { padding: 6px 14px; font-size: 1rem; }
      .info { margin-bottom: 10px; color: #555; }
      .errors { color: red; }
    </style>
  </head>
  <body>
    <h2>Label OCR – {{ fname }}</h2>
    {% if msg %}
      <p class="info">{{ msg }}</p>
    {% endif %}
    {% if error %}
      <p class="errors">{{ error }}</p>
    {% endif %}
    {% if fname %}
      <div class="img-box">
        <img src="{{ url_for('static_img', filename=fname) }}" alt="{{ fname }}">
      </div>
      <form method="post" action="{{ url_for('save_label') }}">
        <input type="hidden" name="filename" value="{{ fname }}">
        <label>Testo letto:</label>
        <input type="text" name="label" value="{{ suggestion }}" autofocus>
        <button class="btn" type="submit">Salva & Avanti</button>
      </form>
      <p style="margin-top:15px;">
        <a href="{{ url_for('skip', filename=fname) }}">Salta questo</a>
      </p>
    {% else %}
      <p>Nessuna immagine da etichettare 🎉</p>
    {% endif %}

    <hr>
    <p><a href="{{ url_for('list_all') }}">Vedi elenco completo</a></p>
  </body>
</html>
"""

LIST_TMPL = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Elenco ROI</title>
    <style>
      body { font-family: sans-serif; margin: 20px; }
      table { border-collapse: collapse; }
      th, td { border: 1px solid #ccc; padding: 4px 6px; }
    </style>
  </head>
  <body>
    <h2>ROI disponibili</h2>
    <table>
      <tr><th>#</th><th>File</th><th>Label</th><th>Azioni</th></tr>
      {% for i, row in rows %}
        <tr>
            <td>{{ i }}</td>
            <td>{{ row["filename"] }}</td>
            <td>{{ row.get("label", "") }}</td>
            <td><a href="{{ url_for('index', fname=row['filename']) }}">Apri</a></td>
        </tr>
      {% endfor %}
    </table>
    <p><a href="{{ url_for('index') }}">Torna al labeling</a></p>
  </body>
</html>
"""

# per servire le immagini direttamente da data_capture
@app.route('/img/<path:filename>')
def static_img(filename):
    # usa send_from_directory ma senza import aggiuntivi
    from flask import send_from_directory
    return send_from_directory(CAPTURE_DIR, filename)

def load_existing_labels():
    labels = {}
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row["label"]
    return labels

def save_label_to_csv(filename, label):
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        fieldnames = ["filename", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"filename": filename, "label": label})

def get_next_unlabeled(labels):
    # ritorna il primo file in data_capture che non è nel csv
    for img_path in sorted(CAPTURE_DIR.glob("*.png")):
        if img_path.name not in labels:
            return img_path.name
    for img_path in sorted(CAPTURE_DIR.glob("*.jpg")):
        if img_path.name not in labels:
            return img_path.name
    return None

@app.route("/")
@app.route("/label/")
@app.route("/label/<fname>")
def index(fname=None):
    labels = load_existing_labels()
    msg = request.args.get("msg", "")
    error = request.args.get("error", "")
    if not fname:
        fname = get_next_unlabeled(labels)
    suggestion = ""
    return render_template_string(PAGE_TMPL, fname=fname, msg=msg, error=error, suggestion=suggestion)

@app.route("/save", methods=["POST"])
def save_label():
    filename = request.form.get("filename")
    label = request.form.get("label", "").strip().upper()

    if not filename:
        return redirect(url_for("index", error="Nessun file"))

    # opzionale: filtra caratteri non previsti
    allowed = set(ALPHABET)
    if any(c not in allowed for c in label):
        return redirect(url_for("index", fname=filename, error="Caratteri fuori alfabeto"))

    save_label_to_csv(filename, label)
    return redirect(url_for("index", msg=f"Salvato {filename} → {label}"))

@app.route("/skip/<filename>")
def skip(filename):
    # lo segniamo comunque nel csv con label vuoto? meglio di no.
    return redirect(url_for("index", msg=f"Saltato {filename}"))

@app.route("/list")
def list_all():
    labels = load_existing_labels()
    rows = []
    i = 1
    for img_path in sorted(CAPTURE_DIR.glob("*.png")):
        rows.append( (i, {"filename": img_path.name, "label": labels.get(img_path.name, "")}) )
        i += 1
    for img_path in sorted(CAPTURE_DIR.glob("*.jpg")):
        rows.append( (i, {"filename": img_path.name, "label": labels.get(img_path.name, "")}) )
        i += 1
    return render_template_string(LIST_TMPL, rows=rows)

if __name__ == "__main__":
    # per usare da rete metti host='0.0.0.0'
    app.run(host="0.0.0.0", port=5000, debug=True)
