# 🌌 Universe Cataloger: Galaxy Morphology Classification

## 📝 O projekcie
"Katalogowanie Wszechświata" to projekt z dziedziny Computer Vision (widzenia komputerowego), którego celem jest automatyczna klasyfikacja morfologii galaktyk na podstawie surowych zdjęć z teleskopów. Model oparty jest na dwuwymiarowej Konwolucyjnej Sieci Neuronowej (2D CNN). 

## 🎯 Cele projektu
- Zbudowanie i wytrenowanie od zera architektury 2D CNN w PyTorch.
- Klasyfikacja zdjęć obiektów głębokiego kosmosu zgodnie z sekwencją Hubble'a (galaktyki spiralne, eliptyczne, nieregularne).
- Wdrożenie przetrenowanego modelu do interaktywnej aplikacji webowej (Drag & Drop).

## 🛠️ Tech Stack
- **Język:** Python 3.10+
- **Deep Learning Framework:** PyTorch (z akceleracją CUDA)
- **Data Science:** Pandas, NumPy, Matplotlib, Jupyter Notebooks
- **Środowisko:** Miniconda, VS Code

## 📂 Struktura Repozytorium
```text
katalogowanie_wszechswiata/
├── data/                 # Folder ignorowany przez Git (dane pobierane lokalnie)
├── models/               # Zapisane wagi wytrenowanej sieci (.pth)
├── notebooks/            # Eksperymenty i wizualizacja danych (Jupyter)
├── src/                  # Kod źródłowy (Architektura CNN, Data Loadery, Training Loop)
├── ui/                   # Interfejs użytkownika
└── requirements.txt      # Zależności projektu